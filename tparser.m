% -*- mode: MATLAB; mode: Outline-minor; outline-regexp: "[ ]*\\(%%+\\|function\\|properties\\|methods\\)"; -*-
%% tparser.m: Transition based parser (c) Deniz Yuret, 2014

classdef tparser < matlab.mixin.Copyable

  properties (SetAccess = immutable)	%% Permanent properties, first three user specified during construction
    parser	% the transition system, e.g. @archybrid.
    fselect	% features to be used (see features.m)
    kerparam    % kernel parameters (same as dogma)
    nmove       % number of possible transitions (fn of parser)
    ndims       % dimensionality of feature vectors (fn of fselect and corpus)
    fidx        % end indices of features in ndims (fn of fselect and corpus)
  end
 
  
  properties (SetAccess = public)  	%% Public properties, user should set some of these before parse to effect parsing behavior
    update	% update type, 0 means no update i.e. test mode, default=1
    predict     % probability of following maxscoremove rather than mincostmove for greedy parser, default=1
    average	% use beta2 (averaged coefficients) if true, beta if not, default=~update
    usecache    % whether to use kernel cache, default=true
    beamsize 	% width of the beam for beamparser, default=10
    earlystop   % whether to use earlystop during beam search, default=1
    gpu         % whether to use the gpu, default (gpuDeviceCount>0)
    output      % what to output (a struct with fields feats, score, etc.)
  end
 
  
  properties (SetAccess = private)  	%% Output properties, these will be set by the parser (depending on fields of output)
    feats 	% feature vectors representing parser states
    score       % score of each move
    cost	% cost of each move
    eval        % evaluation metrics
    move        % the moves executed
    head        % the heads predicted
    sidx        % sentence-end indices in move array
    corpus      % last corpus parsed
  end
 
  properties (Constant = true)
    output_fields = strsplit('corpus feats cost score move head sidx eval');
  end
 
  
  properties (SetAccess = private) 	%% Model parameters, can be obtained by training or manually set with set_model_parameters.
    SV          % support vectors
    beta        % final weights after last training iteration
    beta2       % averaged (actually summed) weights
  end
 
  properties (SetAccess = private)
    svtr     
    newsvtr
    newbeta
    newbeta2
    cache    
    cachekeys
    compute
  end
 

  methods (Access = public)             %% Common public methods
  
    function m = tparser(parser, fselect, kerparam, corpus)
      m.parser = parser;
      m.fselect = fselect;
      m.kerparam = kerparam;
      s1 = corpus{1}; % need corpus for dims
      p1 = feval(m.parser, size(s1.wvec, 2));
      m.nmove = p1.NMOVE;
      [f1,m.fidx] = features(p1, s1, m.fselect);
      m.ndims = numel(f1);
    end % tparser
 
 
    function set_model_parameters(m, model)
      m.SV = model.SV;
      m.beta = model.beta;
      m.beta2 = model.beta2;
    end % set_model_parameters
 
     
    function set_cachekeys(m, feats)
      m.cachekeys = feats;
    end % set_feats

  end % methods (Access = public) % common


  methods (Access = private)            %% Common private methods
 
    function score = compute_score(m, x)
      if ~isempty(m.newsvtr)
        m1 = struct('kerparam', m.kerparam, 'average', m.average, 'cache', [],...
                    'svtr', m.newsvtr, 'beta', m.newbeta, 'beta2', m.newbeta2);
        score = compute_kernel(m1, x);
        if ~isempty(m.svtr)
          score = score + compute_kernel(m, x);
        end
      elseif ~isempty(m.svtr)
        score = compute_kernel(m, x);
      else
        score = zeros(m.nmove, size(x, 2));
      end
    end

 
    function initialize_model(m, corpus)
      nwords = 0;
      for i=1:numel(corpus) nwords = nwords + size(corpus{i}.wvec,2); end
      msg('tparser(%dc,%dd,%ds) corpus(%ds,%dw)', m.nmove, m.ndims, size(m.beta,2), numel(corpus), nwords);
      if isempty(m.update) m.update = 1; end
      if isempty(m.predict) m.predict = 1; end % TODO: this should be gparse specific
      if isempty(m.gpu) m.gpu = gpuDeviceCount(); end
      if isempty(m.usecache) m.usecache = true; end

      if ~isfield(m.output,'feats') m.output.feats = false; end
      if ~isfield(m.output,'score') m.output.score = m.update || m.predict; end
      if ~isfield(m.output,'eval') m.output.eval = m.predict && isfield(corpus{1},'head'); end
      if ~isfield(m.output,'cost') m.output.cost = m.update || ~m.predict || m.output.eval; end
      if ~isfield(m.output,'move') m.output.move = true; end
      if ~isfield(m.output,'head') m.output.head = true; end
      if ~isfield(m.output,'sidx') m.output.sidx = true; end
      if ~isfield(m.output,'corpus') m.output.corpus = true; end
      for i=1:numel(m.output_fields) m.(m.output_fields{i}) = []; end

      m.compute.cost = m.output.cost || m.output.eval || m.update || ~m.predict;
      m.compute.score  = m.output.score || m.update || m.predict;
      m.compute.feats = m.output.feats || m.compute.score;

      if m.compute.score
        if isempty(m.average)
          m.average = (~isempty(m.beta2) && ~m.update);
        elseif m.average
          assert(~isempty(m.beta2), 'Please set model.beta2 for averaged model.');
        end
        if isempty(m.SV)
          m.svtr = zeros(0, m.ndims);
          m.beta = zeros(m.nmove, 0);
          m.beta2 = zeros(m.nmove, 0);
        else
          assert(size(m.SV, 1) == m.ndims);
          assert(size(m.SV, 2) == size(m.beta, 2) && size(m.SV, 2) == size(m.beta2, 2));
          assert(size(m.beta, 1) == m.nmove && size(m.beta2, 1) == m.nmove);
          m.svtr = m.SV';
        end
        if m.update
          m.newsvtr = zeros(0, m.ndims);
          m.newbeta = zeros(m.nmove, 0);
          m.newbeta2 = zeros(m.nmove, 0);
        end
        if m.gpu
          assert(gpuDeviceCount()>0, 'No GPU detected.');
          msg('Resetting GPU.');
          gdev = gpuDevice;
          reset(gdev);
          msg('Loading model on GPU.');
          m.svtr = gpuArray(m.svtr);
          m.beta = gpuArray(m.beta);
          m.beta2 = gpuArray(m.beta2);
          if m.update
            m.newsvtr = gpuArray(m.newsvtr);
            m.newbeta = gpuArray(m.newbeta);
            m.newbeta2 = gpuArray(m.newbeta2);
          end
        end
        if m.usecache
          if ~isempty(m.cachekeys)
            tmp=tic; msg('Computing cache scores.');
            scores = compute_score(m, m.cachekeys);
            toc(tmp);tmp=tic;msg('Initializing kernel cache.');
            m.cache = kernelcache(m.cachekeys, scores);
            toc(tmp);msg('done');
          end
          m.cachekeys = [];
        else
          m.cache = [];
        end
      end % if compute.score

      msg('mode: update=%d predict=%g average=%d usecache=%d gpu=%d', ...
          m.update, m.predict, m.average, m.usecache, m.gpu);
      cfields = fieldnames(m.compute); cstr = '';
      for i=1:numel(cfields)
        cstr = [cstr ' ' cfields{i} '=' num2str(m.compute.(cfields{i}))];
      end
      msg('compute: %s', cstr);
      ofields = fieldnames(m.output); ostr = '';
      for i=1:numel(ofields)
        ostr = [ostr ' ' ofields{i} '=' num2str(m.output.(ofields{i}))];
      end
      msg('output:%s', ostr);
    end % initialize_model

  
    function finalize_model(m, corpus)
      if m.output.eval m.eval = eval_tparser(m, corpus); end
      if m.output.corpus m.corpus = corpus; end
      if m.compute.score
        if m.update
          m.SV = [gather(m.svtr); gather(m.newsvtr)]';
          m.beta = [gather(m.beta) gather(m.newbeta)];
          m.beta2 = [gather(m.beta2) gather(m.newbeta2)];
          m1 = struct('SV', m.SV, 'beta', m.beta, 'beta2', m.beta2);
          m.set_model_parameters(compactify(m1));
          clear m1;
          % m.cache = []; % may need this to check stats
          m.newsvtr = [];
          m.newbeta = [];
          m.newbeta2 = [];
        elseif m.gpu
          m.beta = gather(m.beta);
          m.beta2 = gather(m.beta2);
        end
        m.svtr = [];
      end % if m.compute.score
    end % finalize_model

  end % methods (Access = private) % common


  methods (Access = public)             %% Gparse public methods
  
    function gparse(m, corpus)
      initialize_gparse(m, corpus);
      msg('Processing sentences...');
      t0 = tic;
      for snum=1:numel(corpus)
        s = corpus{snum};
        p = feval(m.parser, size(s.wvec,2));
        valid = p.valid_moves();
        while any(valid)
          mycost = []; myscore = [];
          if m.compute.cost
            mycost = p.oracle_cost(s.head);
            if m.output.cost m.cost(:,end+1) = mycost; end
          end
          if m.compute.feats
            frow = features(p, s, m.fselect);
            fcol = frow';
            if m.output.feats m.feats(:,end+1) = fcol; end
            if m.usecache m.cachekeys(:,end+1) = fcol; end
          end
          if m.compute.score
            myscore = compute_score(m, fcol);
            if m.output.score m.score(:,end+1) = myscore; end
          end
          if m.update
            gparse_update_model(m, frow, mycost, myscore);
          end
          mymove = gparse_pick_move(m, valid, mycost, myscore);
          if m.output.move m.move(end+1) = mymove; end
          p.transition(mymove);
          valid = p.valid_moves();
        end % while 1
        if m.output.head m.head{end+1} = p.head; end
        if m.output.sidx m.sidx(end+1) = numel(m.move); end
        tock(snum, numel(corpus), t0);
      end % for snum=1:numel(corpus)
      finalize_gparse(m, corpus);
    end % gparse

  end


  methods (Access = private)            %% Gparse private methods
 
    function initialize_gparse(m, corpus)
      initialize_model(m, corpus);
    end % initialize_gparse

 
    function finalize_gparse(m, corpus)
      finalize_model(m, corpus);
    end % finalize_gparse

 
    function move = gparse_pick_move(m, valid, cost, score)
      if rand <= m.predict
        [~,move] = max(score);
        if ~valid(move)
          % TODO: we could also choose mincostmove here
          zscore = score;
          zscore(~valid) = -inf;
          [~,move] = max(zscore);
        end
      else
        [~,move] = min(cost);
      end
    end % gparse_pick_move

 
    function gparse_update_model(m, frow, cost, score)
      m.beta2 = m.beta2 + m.beta;
      m.newbeta2 = m.newbeta2 + m.newbeta;
      [maxscore, maxscoremove] = max(score);
      [mincost, mincostmove] = min(cost);
      switch m.update
       case 0	% no update, test mode
        error('gparse_update_model: m.update=0');

       case 1   % from vectorparser, better with dynamic oracle
        if (cost(maxscoremove) > mincost)
          m.newsvtr(end+1,:) = frow;
          newbeta = zeros(m.nmove, 1);
          newbeta(mincostmove) = 1;
          newbeta(maxscoremove) = -1;
          m.newbeta(:,end+1) = newbeta;
          m.newbeta2(:,end+1) = newbeta;
        end

       case 2   % from perceptron, better with static oracle
        score2 = score;
        score2(mincostmove) = -inf;
        [maxscore2, maxscoremove2] = max(score2);
        % This will update sometimes even when the maxscoremove has
        % mincost but is not the lowest index mincost move.
        if (maxscore2 >= score(mincostmove));
          m.newsvtr(end+1,:) = frow;
          newbeta = zeros(m.nmove, 1);
          newbeta(mincostmove) = 1;
          newbeta(maxscoremove2) = -1;
          m.newbeta(:,end+1) = newbeta;
          m.newbeta2(:,end+1) = newbeta;
        end
      end % switch m.update

    end % gparse_update_model

  end % methods (Access = private) % gparse


  methods (Access = public)             %% Bparse public methods
 
    function bparse(m, corpus)

      initialize_bparse(m, corpus);
      t0 = tic;

      for snum=1:numel(corpus)

        sentence = corpus{snum};
        d = 1;
        nbeam = 1;
        m.beam(d,1).sumscore = 0;
        m.beam(d,1).ismincost = true;
        m.beam(d,1).parser = feval(m.parser, size(sentence.wvec,2));
        m.nbeam(d) = nbeam;
        if m.update mincoststate = m.beam(d,1); end
        m.nagenda = 0;

        while any(m.beam(d,1).parser.valid_moves())
          
          % Here is which fields/variables have valid values at each point:
          % +prev +lastmove +sumscore +ismincost +parser -cost -feats -score +mincoststate -agenda -fmatrix
          % Note prev and lastmove are empty for initial state.

          % Check for early stop:

          if m.earlystop 
            found_mincost = false;
            for c = 1:nbeam
              if m.beam(d,c).ismincost
                found_mincost = true;
                break;
                end
            end
            if ~found_mincost break; end
          end
          
          % Set cost and features, fill fmatrix:

          for c = 1:nbeam
            if m.compute.cost
              m.beam(d,c).cost = m.beam(d,c).parser.oracle_cost(sentence.head);
            end
            if m.compute.feats
              m.beam(d,c).feats = features(m.beam(d,c).parser, sentence, m.fselect);
              if m.usecache
                m.cachekeys(:,end+1) = m.beam(d,c).feats;
              end
              if m.compute.score
                fmatrix(:,c) = m.beam(d,c).feats;
              end
            end
          end
          % +prev +lastmove +sumscore +ismincost +parser +cost +feats -score +mincoststate -agenda +fmatrix

          % Computing scores in bulk is faster on gpu.

          if m.compute.score
            scores = compute_score(m, fmatrix(:,1:nbeam));
            for c = 1:nbeam
              m.beam(d,c).score = scores(:,c);
            end
          end
          % +prev +lastmove +sumscore +ismincost +parser +cost +feats +score +mincoststate -agenda -fmatrix

          % Refill agenda with children of beam.

          a = 0;
          for c = 1:nbeam
            cc = m.beam(d,c);
            valid = cc.parser.valid_moves();
            if cc.ismincost mincost=min(cc.cost); end
            for move = 1:m.nmove
              if ~valid(move) continue; end;
              a = a+1;
              m.agenda(a).prev = cc;
              m.agenda(a).lastmove = move;
              if m.predict
                m.agenda(a).sumscore = cc.sumscore + cc.score(move);
              else
                m.agenda(a).sumscore = cc.sumscore - cc.cost(move);
              end
              m.agenda(a).ismincost = (cc.ismincost && (cc.cost(move) == mincost));
            end % for move = 1:m.nmove
          end % for c = 1:nbeam

          m.nagenda = a;
          nbeam = min(m.beamsize, m.nagenda);
          [~, index] = sort([m.agenda(1:m.nagenda).sumscore], 'descend');
          d = d + 1;
          m.nbeam(d) = nbeam;
          for c = 1:nbeam
            ac = m.agenda(index(c));
            m.beam(d,c).prev = ac.prev;
            m.beam(d,c).lastmove = ac.lastmove;
            m.beam(d,c).sumscore = ac.sumscore;
            m.beam(d,c).ismincost = ac.ismincost;
          end

          % +prev +lastmove +sumscore +ismincost -parser -cost -feats -score -mincoststate +agenda -fmatrix

          % Set parser:

          for c = 1:nbeam
            m.beam(d,c).parser = m.beam(d,c).prev.parser.copy();
            m.beam(d,c).parser.transition(m.beam(d,c).lastmove);
          end
          % +prev +lastmove +sumscore +ismincost +parser -cost -feats -score -mincoststate +agenda -fmatrix

          % Track mincoststate; maxscorestate is already in beam(d,1)

          if m.update
            mincoststate = bparse_find_mincoststate(m,d); % uses beam and agenda
          end
          % +prev +lastmove +sumscore +ismincost +parser -cost -feats -score +mincoststate -agenda -fmatrix

        end % while (parse one sentence)

        if (d == 1) error('depth == 1'); end;

        maxscorestate = m.beam(d,1);
        maxscorepath = cell(1,d);
        for i=d:-1:1
          maxscorepath{i} = maxscorestate;
          if (i > 1) maxscorestate = maxscorestate.prev; end;
        end

        bparse_update_dump(m, maxscorepath);

        if m.update
          mincostpath = cell(1,d);
          for i=d:-1:1
            if isempty(mincoststate) error('isempty(mincoststate)'); end;
            mincostpath{i} = mincoststate;
            if (i > 1) mincoststate = mincoststate.prev; end;
          end
          bparse_update_model(m, maxscorepath, mincostpath);
        end
        tock(snum, numel(corpus), t0);

      end % for snum=1:numel(corpus)

      finalize_bparse(m, corpus);
    end % bparse

  end % methods (Access = public) % bparse


  methods (Access = private)            %% Bparse private methods
 
    function initialize_bparse(m, corpus)
      initialize_model(m, corpus);
      if isempty(m.beamsize) m.beamsize = 10; end  % TODO: check gpu to see if this is best
      if isempty(m.earlystop) m.earlystop = true; end
      msg('bparse: beamsize=%d earlystop=%d', m.beamsize, m.earlystop);
      maxlen = 0;
      for i=1:numel(corpus)
        n = size(corpus{i}.wvec, 2);
        if (n > maxlen) maxlen = n; end
      end
      maxdepth = maxlen * 2;
      m.nbeam = zeros(maxdepth);
      m.beam = beamcell;
      m.beam(maxdepth, m.beamsize) = beamcell;
      m.nagenda = 0;
      m.agenda = beamcell;
      m.agenda(m.beamsize * m.nmove) = beamcell;
      m.fmatrix = zeros(m.ndims, m.beamsize);
    end % initialize_bparse

 
    function finalize_bparse(m, corpus)
      finalize_model(m, corpus);
    end % finalize_bparse

 
    function mincoststate = bparse_find_mincoststate(m, d)
      mincoststate = [];
      for c = 1:m.nbeam(d);
        state = m.beam(d,c);
        if state.ismincost
          mincoststate = state;
          break;
        end
      end
      if ~isempty(mincoststate) return; end;
      for c = 1:m.nagenda
        state = m.agenda(c);
        if state.ismincost
          % state.parser is not set in agenda, fix it:
          state.parser = state.prev.parser.copy();
          state.parser.transition(state.lastmove);
          mincoststate = state.copy;
          break;
        end
      end
      if isempty(mincoststate) error('Cannot find mincoststate'); end;
    end % bparse_find_mincoststate

 
    function bparse_update_dump(m, maxscorepath)
      npath = numel(maxscorepath)-1;
      for ipath=1:npath
        s = maxscorepath{ipath};
        if m.output.feats m.feats(:,end+1) = s.feats; end
        if m.output.cost m.cost(:,end+1) = s.cost; end
        if m.output.score m.score(:,end+1) = s.score; end
        if m.output.move m.move(end+1) = maxscorepath{ipath+1}.lastmove; end
      end
      if m.output.head m.head{end+1} = maxscorepath{end}.parser.head; end
      if m.output.sidx m.sidx(end+1) = numel(m.move); end
    end % bparse_update_dump

 
    function bparse_update_model(m, maxscorepath, mincostpath)
      npath = numel(maxscorepath)-1;

      % Note here we are doing sentence averaging, not word averaging
      m.beta2 = m.beta2 + m.beta;
      m.newbeta2 = m.newbeta2 + m.newbeta;

      beta = zeros(m.nmove, 2*npath);
      nbeta = 0;
      for ipath = 1:npath
        maxscorestate = maxscorepath{ipath};
        maxscoremove = maxscorepath{ipath+1}.lastmove;
        mincoststate = mincostpath{ipath};
        mincostmove = mincostpath{ipath+1}.lastmove;
        % TODO: look at ties
        samestate = all(maxscorestate.feats(:) == mincoststate.feats(:));
        samemoves = (maxscoremove == mincostmove);
        if samestate && samemoves
          continue;
        else
          m.newsvtr(end+1,:) = maxscorestate.feats;
          nbeta = nbeta + 1;
          beta(maxscoremove, nbeta) = -1;
          if samestate
            beta(mincostmove, nbeta) = 1;
          else
            m.newsvtr(end+1,:) = mincoststate.feats;
            nbeta = nbeta + 1;
            beta(mincostmove, nbeta) = 1;
          end
        end
      end % for ipath = 1:npath
      if nbeta > 0
        beta = beta(:,1:nbeta);
        m.newbeta = [m.newbeta beta];
        m.newbeta2 = [m.newbeta2 beta];
      end
    end % bparse_update_model

  end % methods (Access = private) % bparse


  properties (SetAccess = private)      %% Bparse private properties
    beam
    nbeam
    agenda
    nagenda
    fmatrix
  end

end % classdef tparser < matlab.mixin.Copyable
