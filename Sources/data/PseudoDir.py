class PseudoDir:
    def __init__(self, name, path, is_dir):
        self.name = name
        self.path = path
        self.is_dir = is_dir

    def __repr__(self):
        return self.name
