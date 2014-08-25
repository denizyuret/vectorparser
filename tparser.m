% tparser.m: Transition based parser (c) Deniz Yuret, 2014

classdef tparser < matlab.mixin.Copyable

% Permanent properties, first three user specified during construction
  properties (SetAccess = immutable)
    parser	% the transition system, e.g. @archybrid.
    fselect	% features to be used (see features.m)
    kerparam    % kernel parameters (same as dogma)
    nmove       % number of possible transitions (fn of parser)
    ndims       % dimensionality of feature vectors (fn of fselect and corpus)
    fidx        % end indices of features in ndims (fn of fselect and corpus)
  end

  % User should set some of these before parse to effect parsing behavior
  properties (SetAccess = public)
    update	% update type, 0 means no update i.e. test mode, default=1
    predict     % probability of following maxscoremove rather than mincostmove for greedy parser, default=1
    average	% use beta2 (averaged coefficients) if true, beta if not, default=~update
    gpu         % whether to use the gpu, default (gpuDeviceCount>0)
    output      % what to output (a struct with fields feats, score, etc.)
    beam        % width of the beam for beamparser, default=10
    earlystop   % whether to use earlystop during beam search, default=1
  end

  % These will be set by the parser (depending on fields of output)
  properties (SetAccess = private)
    feats 	% feature vectors representing parser states
    score       % score of each move
    cost	% cost of each move
    eval        % evaluation metrics
    move        % the moves executed
    head        % the heads predicted
    sidx        % sentence-end indices in move array
    corpus      % last corpus parsed
  end

  % Model parameters, can be obtained by training or manually set with set_model_parameters.
  properties (SetAccess = private)
    SV          % support vectors
    beta        % final weights after last training iteration
    beta2       % averaged (actually summed) weights
  end

  properties (SetAccess = private)  % set Access = private after debugging
    svtr     
    newsvtr
    newbeta
    newbeta2
    cache    
    cachekeys
    compute
    candidates
    agenda
    fmatrix
  end


  methods (Access = public)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function m = tparser(parser, fselect, kerparam, corpus)
      m.parser = parser;
      m.fselect = fselect;
      m.kerparam = kerparam;
      s1 = corpus{1}; % need corpus for dims
      p1 = feval(m.parser, m.sentence_length(s1));
      m.nmove = p1.NMOVE;
      [f1,m.fidx] = features(p1, s1, m.fselect);
      m.ndims = numel(f1);
    end % tparser


    %%%%%%%%%%%%%%%%%%%%%%%%%%
    function gparse(m, corpus)
      initialize_gparse(m, corpus);
      msg('Processing sentences...');
      t0 = tic;
      for snum=1:numel(corpus)
        s = corpus{snum};
        p = feval(m.parser, m.sentence_length(s));
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
          end
          if m.compute.score
            myscore = compute_score(m, fcol);
            if m.output.score m.score(:,end+1) = myscore; end
          end
          if m.update
            perceptron_update(m, frow, mycost, myscore);
          end
          mymove = pick_move(m, valid, mycost, myscore);
          if m.output.move m.move(end+1) = mymove; end
          p.transition(mymove);
          valid = p.valid_moves();
        end % while 1
        if m.output.head m.head{end+1} = p.head; end
        if m.output.move m.sidx(end+1) = numel(m.move); end
        tock(snum, numel(corpus), t0);
      end % for snum=1:numel(corpus)
      if m.output.eval m.eval = eval_model_tparser(m, corpus); end
      if m.output.corpus m.corpus = corpus; end
      finalize_gparse(m, corpus);
    end % gparse


    %%%%%%%%%%%%%%%%%%%%%%%%%
    function bparse(m, corpus)

      initialize_bparse(m, corpus);
      t0 = tic;

      for snum=1:numel(corpus)

        sentence = corpus{snum};
        m.candidates(1).sumscore = 0;
        m.candidates(1).ismincost = true;
        ncandidates = 1;
        nagenda = 0;
        depth = 1;

        while 1
          
          % Here is which fields/variables have valid values at each point:
          % +prev +lastmove +sumscore +ismincost -parser -cost -feats -score -mincoststate +agenda -fmatrix

          % Set parser:

          if isempty(m.candidates(1).prev)
            if (ncandidates ~= 1) error('Not initial state'); end;
            m.candidates(1).parser = feval(m.parser, sentence_length(sentence));
          else
            for c = 1:ncandidates
              m.candidates(c).parser = m.candidates(c).prev.parser.copy();
              m.candidates(c).parser.transition(m.candidates(c).lastmove);
            end
          end

          % Track mincoststate; maxscorestate is already in candidates(1)
          % +prev +lastmove +sumscore +ismincost +parser -cost -feats -score -mincoststate +agenda -fmatrix

          if m.compute.cost
            mincoststate = find_mincoststate(ncandidates, nagenda)
          end

          % Check for early stop:
          % +prev +lastmove +sumscore +ismincost +parser -cost -feats -score +mincoststate -agenda -fmatrix

          if (m.earlystop && ~any(arrayfun(@(x) isfield(x,'ismincost'), m.candidates(1:ncandidates))))
            break
          end
          
          % Check for end of sentence:

          if ~any(m.candidates(1).parser.valid_moves())
            break
          end

          % Set cost and features:
          % +prev +lastmove +sumscore +ismincost +parser -cost -feats -score +mincoststate -agenda -fmatrix

          for c = 1:ncandidates
            if m.compute.cost
              m.candidates(c).cost = m.candidates(c).parser.oracle_cost(sentence.head);
            end
            if m.compute.feats
              m.candidates(c).feats = features(m.candidates(c).parser, sentence, m.fselect)';
              if m.compute.score
                fmatrix(:,c) = m.candidates(c).feats;
              end
            end
          end

          % Computing scores in bulk is faster on gpu.
          % +prev +lastmove +sumscore +ismincost +parser +cost +feats -score +mincoststate -agenda +fmatrix

          if m.compute.score
            scores = compute_kernel(m, fmatrix(:,1:ncandidates));
            for c = 1:ncandidates
              m.candidates(c).score = scores(:,c);
            end
          end

          % Refill agenda with children of candidates.
          % +prev +lastmove +sumscore +ismincost +parser +cost +feats +score +mincoststate -agenda -fmatrix

          depth = depth + 1;
          a = 0;

          for c = 1:ncandidates
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
              if (m.compute.cost && isfield(cc, 'ismincost') && (cc.cost(move) == min(cc.cost)))
                m.agenda(a).ismincost = true;
              end
            end % for move = 1:m.nmove
          end % for c = 1:ncandidates

          nagenda = a;
          ncandidates = min(m.beam, nagenda);
          [~, index] = sort([agenda(1:nagenda).sumscore], 'descend');
          m.candidates(1:ncandidates) = m.agenda(index(1:ncandidates));

          % +prev +lastmove +sumscore +ismincost -parser -cost -feats -score -mincoststate +agenda -fmatrix

        end % while 1 (parse one sentence)

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

      end % for snum=1:numel(corpus)

      finalize_bparse(m, corpus);
    end % bparse


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function set_model_parameters(m, model)
      m.SV = model.SV;
      m.beta = model.beta;
      m.beta2 = model.beta2;
    end % set_model_parameters


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function set_feats(m, feats)
      m.feats = feats; % to debug caching
    end % set_feats

  end % methods (Access = public)

  methods (Access = private)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function initialize_model(m, corpus)
      msg('tparser(%d,%d) corpus(%d)', m.nmove, m.ndims, numel(corpus));
      if isempty(m.update) m.update = 1; end
      if isempty(m.predict) m.predict = 1; end
      if isempty(m.gpu) m.gpu = gpuDeviceCount(); end

      if ~isfield(m.output,'feats') m.output.feats = m.update || m.predict; end
      if ~isfield(m.output,'score') m.output.score = m.update || m.predict; end
      if ~isfield(m.output,'eval') m.output.eval = m.predict && isfield(corpus{1},'head'); end
      if ~isfield(m.output,'cost') m.output.cost = m.update || ~m.predict || m.output.eval; end
      if ~isfield(m.output,'move') m.output.move = true; end
      if ~isfield(m.output,'head') m.output.head = true; end
      if ~isfield(m.output,'sidx') m.output.sidx = true; end
      if ~isfield(m.output,'corpus') m.output.corpus = true; end
      if (m.update && ~isempty(m.feats)) m.cachekeys = m.feats; end
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
      end

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


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function initialize_gparse(m, corpus)
      initialize_model(m, corpus);
      if m.compute.score
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
        if ~isempty(m.cachekeys)
          tmp=tic; msg('Computing cache scores.');
          scores = compute_kernel(m, m.cachekeys);
          toc(tmp);tmp=tic;msg('Initializing kernel cache.');
          m.cache = kernelcache(m.cachekeys, scores);
          toc(tmp);msg('done');
        end
      end
    end % initialize_gparse


    %%%%%%%%%%%%%%%%%%%%%%%%%%
    function finalize_gparse(m, corpus)
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
    end % finalize_gparse


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function n = sentence_length(m, s)
      if isfield(s, 'head')
        n = numel(s.head);
      elseif isfield(s, 'wvec')
        n = size(s.wvec, 2);
      else
        error('Cannot get sentence length');
      end
    end % sentence_length


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function perceptron_update(m, frow, cost, score)
      m.beta2 = m.beta2 + m.beta;
      m.newbeta2 = m.newbeta2 + m.newbeta;
      [maxscore, maxscoremove] = max(score);
      [mincost, mincostmove] = min(cost);
      % TODO: try other update methods
      if cost(maxscoremove) > mincost
        m.newsvtr(end+1,:) = frow;
        newbeta = zeros(m.nmove, 1);
        newbeta(mincostmove) = 1;
        newbeta(maxscoremove) = -1;
        m.newbeta(:,end+1) = newbeta;
        m.newbeta2(:,end+1) = newbeta;
      end % if cost(maxscoremove) > mincost
    end % perceptron_update


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function move = pick_move(m, valid, cost, score)
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
    end % pick_move


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function initialize_bparse(m, corpus)
      initialize_model(m, corpus);
      clear m.candidates m.agenda m.fmatrix;
      m.candidates(m.beam) = empty_state;
      m.agenda(m.beam * m.nmove) = empty_state;
      m.fmatrix = zeros(m.ndims, m.beam);
    end % initialize_bparse


  end % methods (Access = private)

  properties (Constant = true)
    empty_state = ...
        struct('prev', [],...       % previous state
               'lastmove', [],...   % move that led to this state from prev
               'sumscore', [],...   % cumulative score including lastmove
               'ismincost', [],...  % true if state can be on mincostpath
               'parser', [],...     % the parser state
               'cost', [],...       % costs of moves from this state
               'feats', [],...      % feature vector for parser state
               'score', []);        % scores of moves from this state

    output_fields = strsplit('corpus feats cost score move head sidx eval');
  end

end % classdef tparser < matlab.mixin.Copyable
