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

    msg('Initializing...');
    m = vectorparser_init(model, corpus, varargin, nargout)
    msg('Processing sentences...');
    t0 = tic;

    for snum=1:numel(corpus)
        s = corpus{snum};
        h = s.head;
        n = numel(h);
        p = feval(m.parser, n);

        while 1                               % parse one sentence
            valid = p.valid_moves();
            if ~any(valid) break; end

            if m.compute_costs
                cost = p.oracle_cost(h); 		% 1019us
                [mincost, mincostmove] = min(cost);
            end

            if m.compute_features
                f = features(p, s, m.feats);  % 1153us
                ftr = f';                         % f is a row vector, ftr column vector
                m.x(:,end+1) = ftr;
            end

            if m.compute_scores
                score = compute_scores(m, ftr);
                [maxscore, maxscoremove] = max(score); % 925us
            end

            if m.update
                update_model(m, mincostmove);
            end

            if ~m.predict
                execmove = mincostmove;
            elseif valid(maxscoremove)
                execmove = maxscoremove;
            else
                zscore = score;
                zscore(~valid) = -inf;
                [~,execmove] = max(zscore);
            end

            p.transition(execmove);

            if m.dump 
                update_dump();
            end

        end % while 1

        if m.dump
            m.pred{end+1} = p.head;
        end

        dot(snum, numel(corpus), t0);
    end % for s1=corpus

    if m.dump 
        if m.compute_features
            [~,m.fidx] = features(p, s, m.feats);
        end
        dump = m;
    end


    %%%%%%%%%%%%%%%%%%%%%%
    function update_dump()
        if m.compute_costs
            m.y(end+1) = mincostmove;
            m.cost(:,end+1) = cost;
        end
        if m.compute_scores
            m.z(end+1) = execmove;
            m.score(:,end+1) = score;
        end
    end % update_dump


end % vectorparser

%%%%%%%%%%%%%%%%%%%%%%%%%%%
function update_model(m, y)
% TODO: think about minibatching this
    for l=numel(m.net):-1:2
        y = m.net{l}.back(y);
    end
    net{1}.back(y);
    for l=1:numel(m.net)
        m.net{l}.update();
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function score = compute_scores(m, x)
% TODO: optimize single vector multiplication in rnet
% TODO: maybe soft.forw does not need to do softmax
% and we can do back (with softmax) and update in minibatches.
    for l=1:numel(m.net)
        if m.update && m.net{l}.dropout
            x = m.net{l}.drop(x);
        end
        x = m.net{l}.forw(x);
    end
    score = x;
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

    m.dump = (nargout_save >= 2);
    m.compute_costs = m.update || m.dump || ~m.predict; % eval_conll needs costs
    m.compute_features = m.update || m.dump || m.predict;
    m.compute_scores  = m.update || m.predict;

    assert(isfield(m,'parser'), 'Please specify model.parser.');
    tmp_s = corpus{1};
    tmp_p = feval(m.parser, numel(tmp_s.head));
    nc = tmp_p.NMOVE;

    if m.compute_features
        assert(isfield(m,'feats'), 'Please specify model.feats.');
        assert(size(m.feats, 2) == 3, 'The feats matrix needs 3 columns.');
        tmp_f = features(tmp_p, tmp_s, m.feats);
        nd = numel(tmp_f);
    end

    if m.compute_scores
        assert(isfield(m,'net'), 'Please specify model.net.');
    end % if m.compute_scores

    if m.predict
        fprintf('Using predicted moves.\n');
    else
        fprintf('Using gold moves.\n');
    end % if m.predict

    if m.compute_features
        m.x = zeros(nd, 0, 'like', x1);
    end

    if m.dump
        fprintf('Dumping results.\n');
        if m.compute_costs
            m.y = zeros(1, 0, 'like', x1);
            m.cost = zeros(nc, 0, 'like', x1);
        end
        if m.compute_scores
            m.z = zeros(1, 0, 'like', x1);
            m.score = zeros(nc, 0, 'like', x1);
        end
        m.pred = {};
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

