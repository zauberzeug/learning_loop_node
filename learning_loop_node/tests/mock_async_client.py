class MockAsyncClient():
    def __init__(self):
        self.history = []
        
    

    async def call(self,*args, **kwargs):
        self.history.append((args, kwargs))
        return True