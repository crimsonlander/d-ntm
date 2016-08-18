def dec_func(method):
    def wrapper(self, *args, **kwargs):
        print('#'*40)
        return method(self, *args, **kwargs)
    return wrapper


def dec_class(cls):
    attribs = dir(cls)
    for attr in attribs:
        if attr[:2] != '__':
            item = getattr(cls, attr)
            if hasattr(item, '__call__'):
                setattr(cls, attr, dec_func(item))
    return cls

@dec_class
class A(object):
    def __init__(self, x=True):
        if x:
            self.tt = type(self)(False)

    def f(self):
        print("!!!!!")


a = A()
a.f()
a.tt.f()