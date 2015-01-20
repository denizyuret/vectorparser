% Usage: vectorparser takes a model, a corpus, and some options, and
% outputs a model (untouched if update==0), and optionally a dump.
% The dump can be given to eval_conll with the corpus to get some
% statistics.
%
% model = vectorparser(model, corpus);  %% training mode, dump output optional
% [model,dump] = vectorparser(model, corpus, 'update', 0, 'predict', 0); %% dump features
% [model,dump] = vectorparser(model, corpus, 'update', 0)); %% testing
% stats = eval_conll(corpus, dump);

function dump = vectorparser_rnet(model, corpus, varargin)
    msg('Initializing...');
    m = vectorparser_init(model, corpus, varargin, nargout)

    msg('Processing sentences...');
    batches = ceil(numel(corpus)/m.batch);
    output = cell(1, batches);
    t0 = tic;

    parfor batch=1:batches
        output{batch} = parse_batch(m, corpus, batch);
        dot(batch*m.batch, numel(corpus), t0);
    end

    msg('Concatenating output...');
    dump = struct;
    for batch=1:batches
        names = fieldnames(output{batch});
        for i=1:numel(names)
            n = names{i};
            if ~isfield(dump, n)
                dump.(n) = output{batch}.(n);
            else
                dump.(n) = [ dump.(n), output{batch}.(n) ];
            end
        end
    end
end

function o = parse_batch(m, corpus, batch)
    bstart = (batch-1)*m.batch+1;
    bend = min(batch*m.batch, numel(corpus));

    % Initialize a batch of sentences and parsers:
    sentences = corpus(bstart:bend);
    nsentences = numel(sentences);
    parsers = cell(1, nsentences);
    nwords = 0;
    for i=1:nsentences
        iwords = numel(sentences{i}.head);
        parsers{i} = feval(m.parser, iwords);
        nwords = nwords + iwords;
    end
    nmoves = 2 * (nwords - nsentences);

    % Pre-allocate arrays:
    valid = false(m.nmove, nsentences);
    dtype = class(sentences{1}.wvec);
    if m.compute_costs
        cost = zeros(m.nmove, nsentences, dtype);
        o.y = zeros(1, nmoves, dtype);
        o.cost = zeros(m.nmove, nmoves, dtype);
    end
    if m.compute_features
        feats = zeros(m.ndims, nsentences, dtype);
        o.x = zeros(m.ndims, nmoves, dtype);
    end
    if m.compute_scores
        o.z = zeros(1, nmoves, dtype);
        o.score = zeros(m.nmove, nmoves, dtype);
    end

    % Initialize the number and indices of unfinished sentences:
    % the range of sentences, parsers, valid is 1:nsentences.
    % the range of all other variables is idx+1:idx+nvalid.
    % i=ivalid(j) converts from j=1:nvalid to i=1:nsentences.
    ivalid = 1:nsentences;
    nvalid = nsentences;
    j0 = 0;

    while 1  % parse one batch
        %% Update valid moves and unfinished sentences:
        for j=1:nvalid
            i = ivalid(j);
            valid(:,i) = parsers{i}.valid_moves();
        end
        ivalid = find(sum(valid(:,1:nsentences)));
        nvalid = numel(ivalid);
        if nvalid == 0 break; end

        j1 = j0+1;
        j2 = j0+nvalid;
        if m.compute_costs
            for j=1:nvalid
                i = ivalid(j);
                o.cost(:,j0+j) = parsers{i}.oracle_cost(sentences{i}.head);
            end
            [~,o.y(:,j1:j2)] = min(o.cost(:,j1:j2));
        end
        if m.compute_features
            for j=1:nvalid
                i = ivalid(j);
                o.x(:,j0+j) = features(parsers{i}, sentences{i}, m.feats)';
            end
        end
        if m.compute_scores
            o.score(:,j1:j2) = compute_scores(m, o.x(:,j1:j2));
        end
        if ~m.predict
            execmove = o.y;
        else
            zscore = o.score(:,j1:j2);
            zscore(~valid(:,ivalid)) = -inf;
            [~,o.z(:,j1:j2)] = max(zscore);
            execmove = o.z;
        end
        for j=1:nvalid
            i = ivalid(j);
            parsers{i}.transition(execmove(j0+j));
        end
        j0 = j2;
    end % while 1
    for i=1:nsentences
        o.pred{i} = parsers{i}.head;
    end
    if ~m.dumpx
        o = rmfield(o, 'x');
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%
function update_model(m, y)
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
          case 'batch'
            m.batch = v1;
          case 'dumpx'
            m.dumpx = v1;
          otherwise 
            error('Usage: dump = vectorparser_rnet(model, corpus, ...)');
        end
    end

    if ~isfield(m, 'predict')
        m.predict = true;  % predict: use the model for prediction (default), otherwise follow gold moves
    end
    if ~isfield(m, 'dumpx')
        m.dumpx = false;   % do not dump features unless explicitly asked
    end
    if ~isfield(m, 'batch')
        m.batch = 100;
    end

    % m.dump = 0+(nargout_save >= 2);     % needs to be numeric, used as the next index to write
    m.dump = 1; %DBG
    m.update = 0; %DBG
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

