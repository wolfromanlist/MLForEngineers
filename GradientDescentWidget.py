import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import threading

debounce_timer = None

generator = np.random.default_rng(2948292983384)
random_array = 2 * np.linspace(0, 20, 20) + 1 + 10 * (generator.random((1,20)) - 0.5)
x = np.linspace(0, 20, 20)

def mse(y_real, y_predicted):
    return np.squeeze(1/y_real.shape[1] * np.sum((y_real[:, None].T - y_predicted)**2, axis = 0))

def y_hat(w, b, inputs):
    return w * inputs[:,None, None] + b

def my_line(m, b, input_x):
    return m * input_x + b

class GradientDescentWidget():
    def __init__(self, n = 5):
        self.n = n
        self.errors = []
        self.history = []
        self.W, self.B = np.meshgrid(np.linspace(1, 3, 100), np.linspace(0, 2, 100))
        self.Z = mse(random_array, y_hat(self.W, self.B, x))
        self.init_m = 1.0
        self.init_b = 1.0
        self.init_eta = 0.0015
        self.cbar = None 
        self.arrows = []

    def next_step(self, m_now, b_now, lr):
        y_pred = y_hat(m_now, b_now, x)
        error = mse(random_array, y_pred)
        m_grad = -2 * np.mean(x * (random_array - np.squeeze(y_pred)))
        b_grad = -2 * np.mean(random_array - np.squeeze(y_pred))
        m_next = m_now - lr * m_grad
        b_next = b_now - lr * b_grad
        return m_next, b_next, error
    
    def run(self, m_0, b_0, lr, steps):
        self.errors = []
        self.history = []
        self.history.append((m_0, b_0))
        for _ in range(steps - 1):
            if _ == 0:
                m_, b_, error = self.next_step(m_0, b_0, lr)
            else:
                m_, b_, error = self.next_step(m_, b_, lr)
            self.errors.append(error)
            self.history.append((m_, b_))
        self.errors.append(self.next_step(m_, b_, lr)[2])


    def __call__(self):
        fig, ax = plt.subplots(1, 2, figsize = (10,5))
        fig.subplots_adjust(left=0.25, bottom=0.25)
        ax_m = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        ax_b = fig.add_axes([0.25, 0.15, 0.65, 0.03])
        ax_eta = fig.add_axes([0.25, 0.05, 0.65, 0.03])
        ax_c = fig.add_axes([0.93, 0.25, 0.01, 0.65])
        

        m_slider = Slider(
            ax=ax_m,
            label=r'$w_0$',
            valmin=1.0,
            valmax=3.0,
            valinit=self.init_m,
            valstep=None
        )

        b_slider = Slider(
            ax=ax_b,
            label=r'$b_0$',
            valmin=0.0,
            valmax=2.0,
            valinit=self.init_b,
            valstep=None
        )

        eta_slider = Slider(
            ax=ax_eta,
            label='Learning Rate',
            valmin=0.0001,
            valmax=0.0015,
            valinit=self.init_eta,
            valstep=0.0005
        )


        ### calculate error and point history
        self.run(self.init_m, self.init_b, self.init_eta, self.n)

        contours = ax[1].contourf(self.W, self.B, self.Z)
        self.cbar = fig.colorbar(contours, cax = ax_c)
        self.cbar.set_label('Loss')
        ax[1].set_xlabel('W')
        ax[1].set_ylabel('B')
        
        ### draw arrows according to the history
        for i in range(len(self.history) - 1):
                arrow = ax[1].arrow(
                    self.history[i][0], self.history[i][1],
                    self.history[i+1][0] - self.history[i][0],
                    self.history[i+1][1] - self.history[i][1],
                    head_width=0.05, head_length=0.1, fc='red', ec='red'
                )
                self.arrows.append(arrow)  # Store the new arrow in the list

        line, = ax[0].plot(np.linspace(0, self.n, self.n), self.errors, lw=2)
        ax[0].set_xlabel('Step')
        ax[0].set_ylabel('Loss')

        # creating the update function called when manipulating the sliders
        def update(val):
            self.run(m_slider.val, b_slider.val, eta_slider.val, self.n)
            line.set_ydata(self.errors)
            ax[0].set_ylim([0, max(self.errors) + 0.1])
            ax[0].set_xlim([0, self.n])
            ax[0].set_xlabel('Step')
            ax[0].set_ylabel('Loss')
            ax[0].set_title('Loss over time')


            # Remove old arrows
            for arrow in self.arrows:
                arrow.remove()
            self.arrows = []  # Reset the list

            # Add new arrows
            for i in range(len(self.history) - 1):
                arrow = ax[1].arrow(
                    self.history[i][0], self.history[i][1],
                    self.history[i+1][0] - self.history[i][0],
                    self.history[i+1][1] - self.history[i][1],
                    head_width=0.05, head_length=0.1, fc='red', ec='red'
                )
                self.arrows.append(arrow)  # Store the new arrow in the list
            fig.canvas.flush_events() 
            fig.canvas.draw_idle()

        def debounced_update(val):
            global debounce_timer
            if debounce_timer:
                debounce_timer.cancel()  # Cancel previous timer if still running

            debounce_timer = threading.Timer(0.11, update, [val])  # Delay of 0.1s
            debounce_timer.start()

        # linking both sliders to the update function
        m_slider.on_changed(debounced_update)
        b_slider.on_changed(debounced_update)
        eta_slider.on_changed(debounced_update)

        plt.show()