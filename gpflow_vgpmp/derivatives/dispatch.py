from gpflow.utilities import Dispatcher

K_grad = Dispatcher("kernel_first_order_derivative")
magic_Kuu = Dispatcher("kernel_uu_constrained")
magic_Kuf = Dispatcher("kernel_uf_constrained")
K_grad_grad = Dispatcher("kernel_second_order_derivative")
