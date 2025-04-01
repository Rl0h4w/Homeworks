import sys
from collections import defaultdict


def main():
    sys.setrecursionlimit(1 << 25)
    n = int(sys.stdin.readline())
    adj = defaultdict(list)
    for _ in range(n - 1):
        a, b, t = map(int, sys.stdin.readline().split())
        adj[a].append((b, t))
        adj[b].append((a, t))

    def find_furthest(start):
        max_dist = -1
        far_node = start
        stack = [(start, -1, 0)]
        dist = {}
        dist[start] = 0
        while stack:
            node, parent, current_sum = stack.pop()
            for neighbor, t in adj[node]:
                if neighbor != parent:
                    new_sum = current_sum + t
                    if neighbor not in dist or new_sum > dist.get(neighbor, -1):
                        dist[neighbor] = new_sum
                        stack.append((neighbor, node, new_sum))
                        if new_sum > max_dist:
                            max_dist = new_sum
                            far_node = neighbor
        return far_node, max_dist, dist

    v, _, _ = find_furthest(1)
    w, max_d, dist_v = find_furthest(v)

    def get_path(start, end, distance_map):
        parent = {}
        stack = [(start, -1)]
        parent[start] = -1
        found = False
        while stack and not found:
            node, p = stack.pop()
            for neighbor, t in adj[node]:
                if neighbor != p:
                    if distance_map[neighbor] == distance_map[node] + t:
                        parent[neighbor] = node
                        stack.append((neighbor, node))
                        if neighbor == end:
                            found = True
                            break
        path = []
        current = end
        while current != -1:
            path.append(current)
            current = parent.get(current, -1)
        path.reverse()
        return path

    path = get_path(v, w, dist_v)

    sum_v = [dist_v[node] for node in path]
    D_total = sum_v[-1] if path else 0

    min_max = float("inf")
    best_node = -1
    for i in range(len(path)):
        current_sum = sum_v[i]
        current_max = max(current_sum, D_total - current_sum)
        if current_max < min_max:
            min_max = current_max
            best_node = path[i]

    print(best_node)


if __name__ == "__main__":
    main()
