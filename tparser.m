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
    gpu         % whether to use the gpu, default (gpuDeviceCount>0)
    output      % what to output (a struct with fields feats, score, etc.)
    beam        % width of the beam for beamparser, default=10
    earlystop   % whether to use earlystop during beam search, default=1
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
 
     
    function set_feats(m, feats)
      m.feats = feats;
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
      msg('tparser(%d,%d) corpus(%d)', m.nmove, m.ndims, numel(corpus));
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
        end
      end % if compute.score

      msg('mode: update=%d predict=%g average=%d gpu=%d', m.update, m.predict, m.average, m.gpu);
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
      if m.output.eval m.eval = eval_model_tparser(m, corpus); end
      if m.output.corpus m.corpus = corpus; end
      if m.compute.score
        if m.update
          m.SV = [gather(m.svtr); gather(m.newsvtr)]';
          m.beta = [gather(m.beta) gather(m.newbeta)];
          m.beta2 = [gather(m.beta2) gather(m.newbeta2)];
          m1 = struct('SV', m.SV, 'beta', m.beta, 'beta2', m.beta2);
          m.set_model_parameters(compactify(m1));
          clear m1 m.cache m.newsvtr m.newbeta m.newbeta2;
        elseif m.gpu
          m.beta = gather(m.beta);
          m.beta2 = gather(m.beta2);
        end
        clear m.svtr;
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
        m.candidates(1).sumscore = 0;
        m.candidates(1).ismincost = true;
        m.candidates(1).parser = feval(m.parser, size(sentence.wvec,2));
        m.ncandidates = 1;
        mincoststate = m.candidates(1);
        m.nagenda = 0;
        depth = 1;

        while any(m.candidates(1).parser.valid_moves())
          
          % Here is which fields/variables have valid values at each point:
          % +prev +lastmove +sumscore +ismincost +parser -cost -feats -score +mincoststate -agenda -fmatrix
          % Note prev and lastmove are empty for initial state.

          % Check for early stop:

          if (m.earlystop && ~any(arrayfun(@(x)(x.ismincost), m.candidates(1:m.ncandidates))))
            break
          end
          
          % Set cost and features, fill fmatrix:

          for c = 1:m.ncandidates
            if m.compute.cost
              m.candidates(c).cost = m.candidates(c).parser.oracle_cost(sentence.head);
            end
            if m.compute.feats
              m.candidates(c).feats = features(m.candidates(c).parser, sentence, m.fselect)';
              if m.usecache
                m.cachekeys(:,end+1) = m.candidates(c).feats;
              end
              if m.compute.score
                fmatrix(:,c) = m.candidates(c).feats;
              end
            end
          end
          % +prev +lastmove +sumscore +ismincost +parser +cost +feats -score +mincoststate -agenda +fmatrix

          % Computing scores in bulk is faster on gpu.

          if m.compute.score
            scores = compute_score(m, fmatrix(:,1:m.ncandidates));
            for c = 1:m.ncandidates
              m.candidates(c).score = scores(:,c);
            end
          end
          % +prev +lastmove +sumscore +ismincost +parser +cost +feats +score +mincoststate -agenda -fmatrix

          % Refill agenda with children of candidates.

          a = 0;
          for c = 1:m.ncandidates
            cc = m.candidates(c);
            valid = cc.parser.valid_moves();
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
              if (m.compute.cost && cc.ismincost && (cc.cost(move) == min(cc.cost)))
                m.agenda(a).ismincost = true;
              end
            end % for move = 1:m.nmove
          end % for c = 1:m.ncandidates

          m.nagenda = a;
          m.ncandidates = min(m.beam, m.nagenda);
          [~, index] = sort([agenda(1:m.nagenda).sumscore], 'descend');
          m.candidates(1:m.ncandidates) = m.agenda(index(1:m.ncandidates));
          % +prev +lastmove +sumscore +ismincost -parser -cost -feats -score -mincoststate +agenda -fmatrix

          % Set parser:

          for c = 1:m.ncandidates
            m.candidates(c).parser = m.candidates(c).prev.parser.copy();
            m.candidates(c).parser.transition(m.candidates(c).lastmove);
          end
          % +prev +lastmove +sumscore +ismincost +parser -cost -feats -score -mincoststate +agenda -fmatrix

          % Track mincoststate; maxscorestate is already in candidates(1)

          if m.compute.cost
            mincoststate = bparse_find_mincoststate(m); % uses candidates and agenda
          end
          % +prev +lastmove +sumscore +ismincost +parser -cost -feats -score +mincoststate -agenda -fmatrix

          depth = depth + 1;

        end % while (parse one sentence)

        if (depth == 1) error('depth == 1'); end;
        maxscorestate = m.candidates(1);
        maxscorepath = cell(1, depth);
        mincostpath = cell(1, depth);
        while depth > 0
          if m.compute.cost
            if isempty(mincoststate) error('isempty(mincoststate)'); end;
            mincostpath{depth} = mincoststate;
            if (depth > 1) mincoststate = mincoststate.prev; end;
          end
          if m.compute.score
            maxscorepath{depth} = maxscorestate;
            if (depth > 1) maxscorestate = maxscorestate.prev; end;
          end
          depth = depth - 1;
        end % while depth > 0

        bparse_update_dump(m, maxscorepath);
        bparse_update_model(m, maxscorepath, mincostpath);
        tock(snum, numel(corpus), t0);

      end % for snum=1:numel(corpus)

      finalize_bparse(m, corpus);
    end % bparse

  end % methods (Access = public) % bparse


  methods (Access = private)            %% Bparse private methods
 
    function initialize_bparse(m, corpus)
      initialize_model(m, corpus);
      if isempty(m.beam) m.beam = 10; end  % TODO: check gpu to see if this is best
      if isempty(m.earlystop) m.earlystop = true; end
      clear m.candidates m.agenda m.fmatrix;
      m.candidates(m.beam) = empty_state;
      m.ncandidates = 0;
      m.agenda(m.beam * m.nmove) = empty_state;
      m.nagenda = 0;
      m.fmatrix = zeros(m.ndims, m.beam);
    end % initialize_bparse

 
    function finalize_bparse(m, corpus)
      finalize_model(m, corpus);
    end % finalize_bparse

 
    function mincoststate = bparse_find_mincoststate(m)
      mincoststate = [];
      for c = 1:m.ncandidates
        state = m.candidates(c);
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
          mincoststate = state;
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
        samestate = isequal(maxscorestate, mincoststate);
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


  properties (Constant = true)          %% Bparse constant properties
    empty_state = ...
        struct('prev', [],...       % previous state
               'lastmove', [],...   % move that led to this state from prev
               'sumscore', [],...   % cumulative score including lastmove
               'ismincost', [],...  % true if state can be on mincostpath
               'parser', [],...     % the parser state
               'cost', [],...       % costs of moves from this state
               'feats', [],...      % feature vector for parser state
               'score', []);        % scores of moves from this state
  end
 
  properties (SetAccess = private)      %% Bparse private properties
    candidates
    agenda
    ncandidates
    nagenda
    fmatrix
  end

end % classdef tparser < matlab.mixin.Copyable
