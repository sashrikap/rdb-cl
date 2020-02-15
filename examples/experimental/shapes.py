from pyglet.gl import *

window = pyglet.window.Window()

vlist1 = pyglet.graphics.vertex_list(
    3, ("v2f", [0, 0, 400, 50, 200, 300]), ("c3f", [1, 1, 0, 1, 1, 0, 1, 1, 0])
)
vlist2 = pyglet.graphics.vertex_list(
    6, ("v2f", [200, 300, 300, 300, 250, 350, 350, 350, 275, 400, 450, 425])
)


@window.event
def on_draw():
    glClear(pyglet.gl.GL_COLOR_BUFFER_BIT)
    vlist1.draw(GL_TRIANGLES)
    glColor3f(1, 0, 0)
    vlist2.draw(GL_TRIANGLE_STRIP)
    glColor3f(0, 1, 1)
    vlist2.draw(GL_LINE_STRIP)


pyglet.app.run()
