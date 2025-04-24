class DisjointSet:
    def __init__(self):
        self.parent = {} 
        self.rank = {}  

    def find(self, item):
        """Найти корневой элемент множества"""
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 1
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1, item2):
        """Объединить два множества"""
        root1 = self.find(item1)
        root2 = self.find(item2)
        
        if root1 == root2:
            return False
            
        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
        else:
            self.parent[root2] = root1
            if self.rank[root1] == self.rank[root2]:
                self.rank[root1] += 1
                
        return True  # Успешное объединение