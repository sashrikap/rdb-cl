import pyglet


def centered_image(filename):
    img = pyglet.resource.image(filename)
    # import pdb; pdb.set_trace()
    img.anchor_x = img.width / 2
    img.anchor_y = img.height / 2
    return img


class EnvGroups(object):
    def __init__(self):
        self._background = pyglet.graphics.OrderedGroup(0)
        self._road = pyglet.graphics.OrderedGroup(1)
        self._object = pyglet.graphics.OrderedGroup(2)
        self._car = pyglet.graphics.OrderedGroup(3)
        self._main_car = pyglet.graphics.OrderedGroup(4)
        # self._text_background = pyglet.graphics.OrderedGroup(5)
        self._batch = pyglet.graphics.Batch()

    @property
    def batch(self):
        return self._batch

    @property
    def background(self):
        return self._background

    @property
    def road(self):
        return self._road

    @property
    def object(self):
        return self._object

    @property
    def car(self):
        return self._car

    @property
    def main_car(self):
        return self._main_car

    @property
    def text(self):
        return self._text


class TextGroups(object):
    def __init__(self):
        self._text = pyglet.graphics.OrderedGroup(0)
        self._batch = pyglet.graphics.Batch()

    @property
    def text(self):
        return self._text

    @property
    def batch(self):
        return self._batch
