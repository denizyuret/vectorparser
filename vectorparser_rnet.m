% Usage: vectorparser takes a model, a corpus, and some options, and
% outputs a model (untouched if update==0), and optionally a dump.
% The dump can be given to eval_conll with the corpus to get some
% statistics.
%
% model = vectorparser(model, corpus);  %% training mode, dump output optional
% [model,dump] = vectorparser(model, corpus, 'update', 0, 'predict', 0); %% dump features
% [model,dump] = vectorparser(model, corpus, 'update', 0)); %% testing
% stats = eval_conll(corpus, dump);

function [model, dump] = vectorparser_rnet(model, corpus, varargin)
    gpu = gpuDevice; %DBG
    msg('Initializing...');
    m = vectorparser_init(model, corpus, varargin, nargout)
    msg('Processing sentences...');
    t0 = tic;

    for snum1=1:m.batch:numel(corpus)
        snum2=min(numel(corpus), snum1+m.batch-1);
        sentences = corpus(snum1:snum2);
        nsentences = numel(sentences);
        parsers = {};
        for i=1:nsentences
            parsers{i} = feval(m.parser, numel(sentences{i}.head));
        end
        valid = false(m.nmove, nsentences);
        if m.compute_costs
            cost = zeros(m.nmove, nsentences, 'single');
        end
        if m.compute_features
            feats = zeros(m.ndims, nsentences, 'like', sentences{1}.wvec);
        end

        while 1  % parse one batch

            for i=1:nsentences
                valid(:,i) = parsers{i}.valid_moves();
            end
            anyvalid = (sum(valid) > 0);
            if ~any(anyvalid) break; end

            if m.compute_costs
                for i=1:nsentences
                    if anyvalid(i)
                        cost(:,i) = parsers{i}.oracle_cost(sentences{i}.head);
                    end
                end
                [mincost, mincostmove] = min(cost);
            end
            if m.compute_features
                for i=1:nsentences
                    if anyvalid(i)
                        feats(:,i) = features(parsers{i}, sentences{i}, m.feats)';
                    end
                end
            end
            if m.compute_scores
                score = compute_scores(m, feats);
                [maxscore, maxscoremove] = max(score);
            end

            if m.update
                % TODO: make sure finished sentences do not
                % contribute to update.
                error('Not implemented update yet');
                update_model(m, mincostmove);
                wait(gpu); %DBG
            end

            if ~m.predict
                execmove = mincostmove;
            else
                zscore = score;
                zscore(~valid) = -inf;
                [~,execmove] = max(zscore);
            end
            for i=1:nsentences
                if valid(execmove(i), i)
                    parsers{i}.transition(execmove(i));
                end
            end

            if m.dump 
                update_dump();
            end

        end % while 1

        if m.dump
            for i=1:nsentences
                m.pred{snum1+i-1} = parsers{i}.head;
            end
        end

        dot(snum2, numel(corpus), t0);
    end % for s1=corpus

    if m.dump 
        dump = m;
    end


    %%%%%%%%%%%%%%%%%%%%%%
    function update_dump()
        v = (sum(valid) > 0);
        idump1 = m.dump;
        m.dump = m.dump + sum(v);
        idump2 = m.dump - 1;
        if m.compute_features
            m.x(:,idump1:idump2) = feats(:,v);
        end
        if m.compute_costs
            m.y(:,idump1:idump2) = mincostmove(:,v);
            m.cost(:,idump1:idump2) = cost(:,v);
        end
        if m.compute_scores
            m.z(:,idump1:idump2) = execmove(:,v);
            m.score(:,idump1:idump2) = score(:,v);
        end
    end % update_dump


end % vectorparser

%%%%%%%%%%%%%%%%%%%%%%%%%%%
function update_model(m, y)
% TODO: think about minibatching this
    for l=numel(m.net):-1:2
        y = m.net{l}.back(y);
    end
    m.net{1}.back(y);
    for l=1:numel(m.net)
        m.net{l}.update();
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function score = compute_scores(m, x)
% TODO: optimize single vector multiplication in rnet
% TODO: maybe soft.forw does not need to do softmax
% and we can do back (with softmax) and update in minibatches.
% check dbl vs single
    for l=1:numel(m.net)
        if m.update && m.net{l}.dropout
            x = m.net{l}.drop(x);
        end
        x = m.net{l}.forw(x);
    end
    score = gather(x);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function m = vectorparser_init(model, corpus, varargin_save, nargout_save)

    m = model;      % m is a copy of model with extra information
    x1 = corpus{1}.wvec;                % used to determine number type

    for vi = 1:2:numel(varargin_save)
        v = varargin_save{vi};
        v1 = varargin_save{vi+1};
        switch v
          case 'predict' 
            m.predict = v1;
          case 'update'  
            m.update  = v1;
          case 'batch'
            m.batch = v1;
          otherwise 
            error('Usage: [model, dump] = vectorparser(model, corpus, m)');
        end
    end

    if ~isfield(m, 'predict')
        m.predict = true;  % predict: use the model for prediction (default), otherwise follow gold moves
    end
    
    if ~isfield(m, 'update')
        m.update = true;   % update: train the model (default), otherwise model is not updated
    end

    if ~isfield(m, 'batch')
        m.batch = 100;
    end

    m.dump = 0+(nargout_save >= 2);     % needs to be numeric, used as the next index to write
    m.compute_costs = m.update || m.dump || ~m.predict; % eval_conll needs costs
    m.compute_features = m.update || m.dump || m.predict;
    m.compute_scores  = m.update || m.predict;

    assert(isfield(m,'parser'), 'Please specify model.parser.');
    tmp_s = corpus{1};
    tmp_p = feval(m.parser, numel(tmp_s.head));
    m.nmove = tmp_p.NMOVE;

    if m.compute_features
        assert(isfield(m,'feats'), 'Please specify model.feats.');
        assert(size(m.feats, 2) == 3, 'The feats matrix needs 3 columns.');
        [tmp_f,m.fidx] = features(tmp_p, tmp_s, m.feats);
        m.ndims = numel(tmp_f);
    end

    if m.compute_scores
        assert(isfield(m,'net'), 'Please specify model.net.');
    end % if m.compute_scores

    if m.predict
        fprintf('Using predicted moves.\n');
    else
        fprintf('Using gold moves.\n');
    end % if m.predict

    if m.dump
        fprintf('Dumping results.\n');
        nwords = 0;
        for i=1:numel(corpus)
            nwords = nwords + numel(corpus{i}.head);
        end
        nmoves = 2 * (nwords - numel(corpus));
        if m.compute_features
            m.x = zeros(m.ndims, nmoves, 'like', x1);
        end
        if m.compute_costs
            m.y = zeros(1, nmoves, 'like', x1);
            m.cost = zeros(m.nmove, nmoves, 'like', x1);
        end
        if m.compute_scores
            m.z = zeros(1, nmoves, 'like', x1);
            m.score = zeros(m.nmove, nmoves, 'like', x1);
        end
        m.pred = cell(1, numel(corpus));
    end % if m.dump

end % vectorparser_init


%%%%%%%%%%%%%%%%%%%%%%%%%%
function dot(cur, tot, t0)
    t1 = toc(t0);
    if cur == tot
        fprintf('. %d/%d (%.2fs %gx/s)\n', cur, tot, t1, cur/t1);
    elseif mod(cur,10) == 0
        fprintf('.');
        if mod(cur, 100) == 0
            fprintf(' %d/%d (%.2fs %gx/s)\n', cur, tot, t1, cur/t1);
        end
    end
end % dot

