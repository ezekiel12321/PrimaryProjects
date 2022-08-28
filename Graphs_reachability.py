

class Graph(object):

    def __init__(self, vertices=None, edges=None):
        
        self.vertices = set(vertices or [])
        self.edges = set(edges or [])

g = Graph(vertices={'a', 'b', 'c', 'd', 'e', 'f', 'g'},
          edges={('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'd'),
                 ('c', 'a'), ('c', 'e'), ('d', 'b'), ('d', 'c'),
                 ('f', 'g'), ('g', 'f')})


import networkx as nx 

class Graph(object):

    def __init__(self, vertices=None, edges=None):
        
        self.vertices = set(vertices or [])
        self.edges = set(edges or [])

    def show(self):
        g = nx.DiGraph()
        g.add_nodes_from(self.vertices)
        g.add_edges_from(self.edges)
        nx.draw(g, with_labels=True)

g = Graph(vertices={'a', 'b', 'c', 'd', 'e', 'f', 'g'},
          edges={('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'd'),
                 ('c', 'a'), ('c', 'e'), ('d', 'b'), ('d', 'c'),
                 ('f', 'g'), ('g', 'f')})
g.show()


import networkx as nx 

class Graph(object):

    def __init__(self, vertices=None, edges=None):
        
        self.vertices = set(vertices or [])
        self.edges = set(edges or [])

    def show(self):
        g = nx.DiGraph()
        g.add_nodes_from(self.vertices)
        g.add_edges_from(self.edges)
        nx.draw(g, with_labels=True)

    def add_vertex(self, v):
        self.vertices.add(v)

    def add_edge(self, e):
        self.edges.add(e)


import networkx as nx 

class Graph(object):

    def __init__(self, vertices=None, edges=None):
        
        self.vertices = set(vertices or [])
        self.edges = set(edges or [])

    def show(self):
        g = nx.DiGraph()
        g.add_nodes_from(self.vertices)
        g.add_edges_from(self.edges)
        nx.draw(g, with_labels=True)

    def add_vertex(self, v):
        self.vertices.add(v)

    def add_edge(self, e):
        self.edges.add(e)

    def successors(self, u):
        """Returns the set of successors of vertex u"""
        return {v for v in self.vertices if (u, v) in self.edges}

g = Graph(vertices={'a', 'b', 'c', 'd', 'e', 'f', 'g'},
          edges={('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'd'),
                 ('c', 'a'), ('c', 'e'), ('d', 'b'), ('d', 'c'),
                 ('f', 'g'), ('g', 'f')})
g.successors('a')


import networkx as nx 

class Graph(object):

    def __init__(self, vertices=None, edges=None):
        self.s = {u: set() for u in vertices or []}
        for u, v in (edges or []):
            self.add_edge((u, v))

    def show(self):
        g = nx.DiGraph()
        g.add_nodes_from(self.s.keys())
        g.add_edges_from([(u, v) for u in self.s for v in self.s[u]])
        nx.draw(g, with_labels=True)

    def add_vertex(self, v):
        if v not in self.s:
            self.s[v] = set()

    def add_edge(self, e):
        u, v = e
        self.add_vertex(u)
        self.add_vertex(v)
        self.s[u].add(v)

    @property
    def vertices(self):
        return set(self.s.keys())

    def successors(self, u):
        """Returns the set of successors of vertex u"""
        return self.s[u]

g = Graph(vertices={'a', 'b', 'c', 'd', 'e', 'f', 'g'},
          edges={('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'd'),
                 ('c', 'a'), ('c', 'e'), ('d', 'b'), ('d', 'c'),
                 ('f', 'g'), ('g', 'f')})
g.show()
print(g.successors('a'))

"""



The algorithm keeps two sets of vertices: 

* The set of _open_ vertices: these are the vertices that are known to be reachable, and whose successors have not yet been explored. 
* The set of _closed_ vertices: these are the vertices that are known to be reachable, and whose successors we have already explored. 

"""

def reachable(g, v):
    """Given a graph g, and a starting vertex v, returns the set of states
    reachable from v in g."""
    vopen = {v}
    vclosed = set()
    while len(vopen) > 0:
        u = vopen.pop()
        vclosed.add(u)
        vopen.update(g.successors(u) - vclosed)
    return vclosed

print(reachable(g, 'a'))
print(reachable(g, 'g'))

"""To visualize the algorithm, let us write a version where at each iteration, open vertices are drawn in red and closed ones in green"""

def reachable(g, v):
    """Given a graph g, and a starting vertex v, returns the set of states
    reachable from v in g."""
    vopen = {v}
    vclosed = set()
    while len(vopen) > 0:
        u = vopen.pop()
        vclosed.add(u)
        vopen.update(g.successors(u) - vclosed)
    return vclosed

reachable(g, 'a')

"""


"""

def breath_first(g, v):
    """Given a graph g, and a starting vertex v, returns the set of states
    reachable from v in g."""
    
    
    vopen = [v]
    vclosed = set()
    while len(vopen) > 0:
        u = vopen.pop(0) 
        vclosed.add(u)
        
        for w in g.successors(u) - vclosed:
            if w not in vopen:
                vopen.append(w) 
    return vclosed

gg = Graph(vertices={},
           edges={('a', 'b'), ('b', 'c'), ('c', 'd'),
                  ('a', 'u'), ('u', 'v'), ('v', 'w'), ('u', 'z')})

breath_first(gg, 'a')

def depth_first(g, v):
    """Given a graph g, and a starting vertex v, returns the set of states
    reachable from v in g."""
    
    
    vopen = [v]
    vclosed = set()
    while len(vopen) > 0:
        u = vopen.pop() 
        vclosed.add(u)
        
        for w in g.successors(u) - vclosed:
            if w not in vopen:
                vopen.append(w)
    return vclosed

depth_first(gg, 'a')



def graph_edges(self):
    """Yields the edges of the graph, one by one.  Each edge is yielded as a
    pair of vertices (source, destination). """
    
    for x in self.vertices:
        
        for y in self.successors(x):
            
            yield x,y



Graph.edges = property(graph_edges)



"""Here are some tests. """



e = [(1, 2), (1, 3), (2, 3)]
g = Graph(vertices=[1, 2, 3], edges=e)
assert set(g.edges) == set(e)

import types

assert isinstance(g.edges, types.GeneratorType)

"""Here are some randomized test."""



import random

for _ in range(10):
    num_vertices = random.randint(4, 10)
    num_edges = random.randint(1, num_vertices * num_vertices)
    vertices = random.sample(range(0, 1000), num_vertices)
    edges = {(random.choice(vertices), random.choice(vertices)) for _ in range(num_edges)}
    g = Graph(vertices=vertices, edges=edges)
    assert set(g.edges) == edges



def is_tree(g):
    """Returns True iff the graph is a tree."""
    
    
    
    Dict = {}
    for u,v in g.edges:
        if(Dict.get(v) == None):
            Dict[v] = 0
        Dict[v] = Dict[v] + 1
        
    for x in Dict:
        
        
        if Dict[x] > 1:
            return False
    
    counter = 0
    
    for x in g.vertices:
        if x not in Dict:
            counter += 1
    if counter > 1:
        return False

    return True


g = Graph(vertices=[1, 2, 3,4,5,6], edges=[(1, 2), (1, 3),(4,5),(4,6)])
assert not is_tree(g)
g=Graph(vertices=[1,2],edges=[])
assert not is_tree(g)



g = Graph(vertices=[1, 2, 3], edges=[(1, 2), (1, 3)])
assert is_tree(g)
g = Graph(vertices=[1, 2, 3], edges=[(1, 2), (2, 3), (1, 3)])
assert not is_tree(g)
g = Graph(vertices=[1, 2, 3], edges=[(1, 3), (2, 3)])
assert not is_tree(g)



g = Graph()
assert is_tree(g)

def can_reach(v, g1, g2, w):
    """Given two graphs g1, g2 that share the same vertices, and two verteces v, w, 
    returns True if you can go from v to w using edges of either g1 or g2 (mixed any
    way you want) and False otherwise."""
    vopen = {v}
    vclosed = set()
    while len(vopen) > 0:
        u = vopen.pop()
        vclosed.add(u)
        vopen.update(g1.successors(u) - vclosed)
        vopen.update(g2.successors(u) - vclosed)

    if w not in vclosed:
        return False
    return True
    





vertices = {1, 2, 3, 4, 5, 6, 7}
g1 = Graph(vertices=vertices, edges=[(1, 2), (3, 4)])
g2 = Graph(vertices=vertices, edges=[(2, 3), (4, 5), (6, 7)])
assert can_reach(1, g1, g2, 2)
assert can_reach(1, g1, g2, 3)
assert can_reach(1, g1, g2, 4)
assert can_reach(1, g1, g2, 5)
assert not can_reach(1, g1, g2, 6)
assert not can_reach(1, g1, g2, 7)



vertices = set(range(100))

g1 = Graph(vertices=vertices, edges=[(n, 2 * n) for n in range(100) if 2 * n < 100])
g2 = Graph(vertices=vertices, edges=[(n, 3 * n) for n in range(100) if 3 * n < 100])
assert can_reach(1, g1, g2, 6)
assert can_reach(1, g1, g2, 24)
assert can_reach(1, g1, g2, 32)
assert can_reach(1, g1, g2, 9)
assert not can_reach(1, g1, g2, 15)
assert not can_reach(1, g1, g2, 60)
assert can_reach(5, g1, g2, 15)
assert can_reach(5, g1, g2, 30)

