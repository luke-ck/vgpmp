from gpflow.utilities import Dispatcher

K_grad = Dispatcher("kernel_first_order_derivative")
K_grad_grad = Dispatcher("kernel_second_order_derivative")
