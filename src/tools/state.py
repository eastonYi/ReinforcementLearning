class State():
    num_Instances = 0
    list_hash = []
    def __init__(self, data):
        self.data = tuple(data)
        self.id = '_'.join(map(str, self.data))
        self.idx = self.__class__.num_Instances
        if hash(tuple(data)) not in self.__class__.list_hash:
            self.__class__.list_hash.append(hash(tuple(data)))
            self.__class__.num_Instances += 1

    def __str__(self):

        return self.id

    def allowedActions(self, actions):
        
        return actions
