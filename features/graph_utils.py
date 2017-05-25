class UnionFind:
    def __init__(self):
        self.parent = {}
        self.size = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1
        if self.parent[x] == x:
            return x
        else:
            self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

    def same(self, x, y):
        return self.find(x) == self.find(y)

    def size(self, x):
        return self.size[self.find(x)]

    def unite(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x != y:
            self.parent[x] = y
            self.size[y] += self.size[x]
            self.size[x] = 0
