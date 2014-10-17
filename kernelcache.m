classdef kernelcache < handle

    properties
        size;
        keys;
        vals;
        rvec;
        mean;
        std;
        hit;
        miss;
        nkeys;
    end % properties

    methods

        function c = kernelcache(keys, vals);
            ndims = size(keys, 1);
            nkeys = size(keys, 2);

            % The empty cells only take 8 bytes each
            % So this is useless to limit memory:
            % c.size = floor(min(100 * nkeys, 1e11/(bytes(keys(:,1))+bytes(vals(:,1)))));

            c.size = 100 * nkeys;
            c.keys = cell(1, c.size);
            c.vals = cell(1, c.size);
            rng(1);  % for reproducible random numbers
            c.rvec = rand(1, ndims);

            % Do all internal calculations in double
            % Otherwise we get bad cache performance
            normval = c.rvec * double(keys);
            c.mean = mean(normval);
            c.std = std(normval);

            % This works fast but gives slightly different indices
            % with single values compared to doing one at a time.
            % However the rate is 1e-6 with internal calcs in double.
            cnum = cellnum(c, keys, normval);

            c.nkeys = 0;
            for i=1:nkeys
                % j = cellnum(c, keys(:,i));
                j = cnum(i);
                if isempty(c.keys{j}) c.nkeys = c.nkeys + 1; end;
                c.keys{j} = keys(:,i);
                c.vals{j} = vals(:,i);
            end
            c.hit = 0;
            c.miss = 0;
        end

        % The dot product of any key with random vector rvec is normally
        % distributed with c.mean and c.std.  Use normcdf to get a
        % uniformly distributed number in (0,1) from a normally
        % distributed number N(c.mean,c.std).  Convert that to an
        % index by multiplying with c.size and taking ceil.

        function cnum = cellnum(c, keys, normval)
            if nargin < 3
                normval = c.rvec * double(keys);
            end
            unifval = normcdf(normval, c.mean, c.std);
            cnum = ceil(c.size * unifval);
        end

        function val = get(c, key)
            cnum = cellnum(c, key);
            ckey = c.keys{cnum};
            if (~isempty(ckey) && all(ckey == key))
                val = c.vals{cnum};
                c.hit = c.hit + 1;
            else
                val = [];
                c.miss = c.miss + 1;
            end
        end

    end % methods

end % classdef
