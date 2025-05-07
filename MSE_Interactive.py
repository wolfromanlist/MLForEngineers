import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider

generator = np.random.default_rng(2948292983384)
random_array = 2 * np.linspace(0, 20, 20) + 1 + 10 * (generator.random((1,20)) - 0.5)
x = np.linspace(0, 20, 20)

def mse(y_real, y_predicted):
    return 1/y_real.shape[1] * np.sum((y_real[:, None].T - y_predicted)**2, axis = 0)

def y_hat(w, b, inputs):
    return w * inputs[:,None, None] + b

def my_line(m, b, input_x):
    return m * input_x + b

class InteractivePlot():
    def __init__(self, n = 5):
        self.n = n
        self.increment = 2/(n-1)
        self.W, self.B = np.meshgrid(np.linspace(1, 3, self.n), np.linspace(0, 2, self.n))
        self.Z = mse(random_array, y_hat(self.W, self.B, x))
        self.grid = list(zip(self.W.flatten(), self.B.flatten(), self.Z.flatten()))
        self.minimum, self.maximum = np.min(self.Z), np.max(self.Z)
        # creating the figure and the sliders
        self.init_m = 1.0 + self.increment/2
        self.init_b = 0.0 + self.increment/2
        self.point = None
        self.scatter = None
        self.bounds_list = [0,1]
        self.colorbar = None
        self.viridis = mpl.colormaps['viridis']

    def get_entry(self, x, y):
        for entry in self.grid:
            norm = np.linalg.norm(np.array([abs(x - entry[0]), abs(y - entry[1])]), ord = np.inf)
            if norm <= self.increment and x > entry[0] and y > entry[1]:
                return entry
        

    def make_patch(self, ax, anchor, error):
        normed_color = (error - self.minimum)/(self.maximum - self.minimum)
        rect = Rectangle(anchor, self.increment, self.increment, color=self.viridis(normed_color))
        ax.add_patch(rect)


    # Creating random points
    generator = np.random.default_rng(2948292983384)
    random_array = 2 * np.linspace(0, 20, 20) + 1 + 10 * (generator.random((1,20)) - 0.5)
    x = np.linspace(0, 20, 20)

    @staticmethod
    def mse(y_real, y_predicted):
        return 1/y_real.shape[1] * np.sum((y_real - y_predicted)**2)

    def __call__(self):
        fig, ax = plt.subplots(1, 2, figsize = (10,5))
        line, = ax[0].plot(x, my_line(self.init_m, self.init_b, x), lw=2)
        self.point = ax[1].scatter(self.init_m, self.init_b, marker = 'x', color = 'black', zorder =10, alpha = 0.1)
        self.scatter = ax[0].scatter(x, random_array)
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
        ax[1].set_xticks(np.linspace(1, 3, self.n), minor = True)
        ax[1].set_yticks(np.linspace(0, 2, self.n), minor = True)
        ax[1].set_xticks(np.linspace(1, 3, 5))
        ax[1].set_yticks(np.linspace(0, 2, 5))
        ax[1].grid(which = 'minor')
        ax[1].set_xlabel('w')
        ax[1].set_ylabel('b')
        ax[1].set_xlim([1, 3])
        ax[1].set_ylim([0, 2])

        fig.subplots_adjust(left=0.25, bottom=0.25)
        ax_m = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        ax_b = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
        ax_c = fig.add_axes([0.92, 0.25, 0.0225, 0.63])

        self.bounds_list = [0,1]
        self.colorbar = None

        m_slider = Slider(
            ax=ax_m,
            label='Parameter w',
            valmin=1.0 + self.increment/2,
            valmax=3.0 - self.increment/2,
            valinit=self.init_m + self.increment/2,
            valstep=self.increment
        )

        b_slider = Slider(
            ax=ax_b,
            label='Parameter b',
            valmin=0.0 + self.increment/2,
            valmax=2.0 - self.increment/2,
            valinit=self.init_b + self.increment/2,
            valstep=self.increment,
            orientation="vertical"
        )

        # creating the update function called when manipulating the sliders
        def update(val):
            self.point.remove()
            new_y = my_line(m_slider.val, b_slider.val, x)
            line.set_ydata(new_y)
            my_tuple = self.get_entry(m_slider.val, b_slider.val)
            mse_var = InteractivePlot.mse(random_array, new_y)
            ax[0].set_title(f'MSE: {mse_var:.2f}')
            if not mse_var in self.bounds_list:
                self.bounds_list.append(mse_var)
                self.bounds_list = sorted(self.bounds_list)
            self.point = ax[1].scatter(m_slider.val, b_slider.val, marker = 'x', color = 'black', zorder =10)
            self.make_patch(ax[1], (my_tuple[0], my_tuple[1]), mse_var)
            norm = mpl.colors.BoundaryNorm(self.bounds_list, self.bounds_list[-1]/self.maximum * self.viridis.N)
            ax_c.clear()
            self.colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=self.viridis), cax=ax_c)
            step = 1
            if len(self.bounds_list) > 5:
                step = 2
            elif len(self.bounds_list) > 20:
                step = 5
            self.colorbar.set_ticks(self.bounds_list[::step], labels = [f'{b:.2f}' for b in self.bounds_list[::step]])
            
            fig.canvas.draw_idle()

        # linking both sliders to the update function
        m_slider.on_changed(update)
        b_slider.on_changed(update)

        # adding the reset button
        resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', hovercolor='0.975')

        def reset(event):
            m_slider.reset()
            b_slider.reset()
            new_y = my_line(m_slider.val, b_slider.val, x)
            line.set_ydata(new_y)
            ax[1].clear()
            self.bounds_list = [0,1]
            ax_c.clear()
            self.point = ax[1].scatter(self.init_m, self.init_b, marker = 'x', color = 'black', zorder =10, alpha = 0.1)
            ax[1].set_xticks(np.linspace(1, 3, self.n), minor = True)
            ax[1].set_yticks(np.linspace(0, 2, self.n), minor = True)
            ax[1].set_xticks(np.linspace(1, 3, 5))
            ax[1].set_yticks(np.linspace(0, 2, 5))
            ax[1].grid(which = 'minor')
            ax[1].set_xlim([1, 3])
            ax[1].set_ylim([0, 2])
            ax[1].set_xlabel('w')
            ax[1].set_ylabel('b')

            fig.canvas.draw_idle()

            button.ax.figure.canvas.draw()

            
        button.on_clicked(reset)

        plt.show()