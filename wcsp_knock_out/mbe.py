from bucket import Bucket


def mbe_solve(dom_size, all_functions, mem_bound=int(1e7)):
    all_vars = list(dom_size.keys())
    all_buckets = [Bucket.from_matrix(*x) for x in all_functions]
    buckets = dict()
    assign = dict()
    for i, var in enumerate(all_vars):
        related_buckets = [x for x in all_buckets if var in x.dims]
        buckets[var] = [x.copy() for x in related_buckets]
        sizes = [x.data.numel() for x in related_buckets]
        indexes = sorted(range(len(sizes)), key=sizes.__getitem__)
        partition = []
        current_partition = []
        cp_dims = set()
        for idx in range(len(indexes)):
            tmp_cp_dims = set(cp_dims)
            tmp_cp_dims.update(related_buckets[indexes[idx]].dims)
            numel = 1
            for dim in tmp_cp_dims:
                numel *= dom_size[dim]
            if numel > mem_bound:
                partition.append(current_partition)
                current_partition = []
                cp_dims = set()
            current_partition.append(related_buckets[indexes[idx]])
            cp_dims.update(related_buckets[indexes[idx]].dims)
        partition.append(current_partition)
        assert sum([len(x) for x in partition]) == len(related_buckets)
        if i < len(all_vars) - 1:
            for p in partition:
                bucket = Bucket.join(p).proj(var)
                all_buckets.append(bucket)
            for bucket in related_buckets:
                all_buckets.remove(bucket)
        else:
            bucket = Bucket.join(related_buckets)
            assign[var] = bucket.data.argmin().item()
    for i in range(len(all_vars) - 2, -1, -1):
        cur_var = all_vars[i]
        related_buckets = buckets[cur_var]
        related_buckets = [x.reduce(assign) for x in related_buckets]
        bucket = Bucket.join(related_buckets)
        assign[cur_var] = bucket.data.argmin().item()
    cost = 0
    for func, var1, var2 in all_functions:
        cost += func[assign[var1]][assign[var2]]
    return cost


if __name__ == '__main__':
    from core.parser import parse
    vars, functions = parse('../problems/valid/1.xml')
    ds = {k[0]: k[1] for k in vars}
    print(mbe_solve(ds, functions))