def dict_to_vec(dict_var):
    """
    Convert dictionary variable to vector
    """
    vec, dims, shapes = [], {}, {}
    curr_dim = 0
    next_dim = curr_dim
    for key, val in dict_var.items():
        next_dim += jnp.prod(val.shape)
        dims[key] = (curr_dim, next_dim)
        shapes[key] = val.shape
        vec.append(val.flatten())
        curr_dim = next_dim
    vec = jnp.concatenate(vec)
    return vec, dims, shapes


def vec_to_dict(vec, dims, shapes):
    dict_var = {}
    for key, dim in dims.items():
        shape = shapes[key]
        dict_var[key] = vec[dim[0] : dim[1]].reshape(shape)
    return dict_var


class OpenLoopOptimizer(object):
    def __init__(self, f_dyn, f_cost, udim, horizon):
        """
        Equations
        x_{t+1} = f_dyn(x_{t}, u_{t})
        c_{t}   = f_cost(x_{t}, u_{t}, x_{t+1})

        Params
        : f_dyn    : forward dynamics
        : f_cost   : cost function
        : udim     : dimension of action
        : horizon  : time horizon
        """
        self.f_cost = f_cost
        self.f_dyn = f_dyn
        self.udim = udim
        self.horizon = horizon

    def optimize_x0(self, x0, u0):
        raise NotImplementedError

    def optimize_u(self, x0):
        raise NotImplementedError


class LocalOptimizer(OpenLoopOptimizer):
    """ L-BFGS based optimizer """

    def __init__(self, f_dyn, f_cost, udim, horizon, jit=False):
        super().__init__(f_dyn, f_cost, udim, horizon)
        # Optimize reward based on JAX & scipy.optim
        self.grad_xu = jax.grad(self.cost_xu)
        self.hessian_xu = jax.hessian(self.cost_xu)
        self.jacobian_xu = jax.jacfwd(self.grad_xu)
        if jit:
            self.grad_xu = jax.jit(self.grad_xu)
            self.hessian_xu = jax.jit(self.hessian_xu)
            self.jacobian_xu = jax.jit(self.jacobian_xu)

    # @partial(jax.jit, static_argnums=(0,))
    def cost_xu(self, xu):
        """
        Param:
        : xu : concatenate([x, u])
        """
        cost = 0.0
        x, u = self.divide_xu(xu)
        for t in range(self.horizon):
            x_dict = self.vec_to_dict(x)
            next_x_dict = self.f_dyn(x_dict, u[t])
            cost += self.f_cost(x_dict, u[t], next_x_dict)
            x, _, _ = self.dict_to_vec(next_x_dict)
        return cost

    def concate_xu(self, x, u):
        """ Computational graph input: x, u => [x, u.flatten()] """
        xu = jnp.concatenate([x, u.flatten()])
        return xu

    def divide_xu(self, xu):
        """ Computational graph output: [x, u.flatten()] => x, u """
        return xu[: -self.udim], xu[-self.udim :].reshape(-1, self.udim)

    def to_numpy(self, arr):
        """ To original numpy array """
        return onp.array(arr).astype(onp.float64)

    def dict_to_vec(self, dict_var):
        """
        Convert dictionary variable to vector
        """
        vec, dims, shapes = [], {}, {}
        curr_dim = 0
        next_dim = curr_dim
        for key, val in dict_var.items():
            next_dim += jnp.prod(val.shape)
            dims[key] = (curr_dim, next_dim)
            shapes[key] = val.shape
            vec.append(val.flatten())
            curr_dim = next_dim
        vec = jnp.concatenate(vec)
        return vec, dims, shapes

    def vec_to_dict(self, vec):
        dict_var = {}
        for key, dim in self.dims.items():
            shape = self.shapes[key]
            dict_var[key] = vec[dim[0] : dim[1]].reshape(shape)
        return dict_var

    def optimize_u(self, x0, u0):
        """
        Param
        : x0 : dict, initial state
        : u0 : array, initial guess of action
        """
        # Numpy-based grad (scipy.optim needs float64)
        x0, dims, shapes = self.dict_to_vec(x0)
        self.shapes = shapes
        self.dims = dims

        def grad_u(u):
            xu = self.concate_xu(x0, u)
            return self.to_numpy(self.grad_xu(xu))[-self.udim :]

        # Numpy-based reward
        def cost_u(u):
            xu = self.concate_xu(x0, u)
            return float(self.cost_xu(xu))

        # Optimize
        u0 = u0.flatten()
        umax, rmax, info = fmin_l_bfgs_b(cost_u, u0, grad_u)
        umax = umax.reshape(-1, self.udim)
        u_info = {"du": info["grad"]}
        return umax, rmax, u_info

    def optimize_x0(self, x0, u0, invert=False):
        """
        Optimize environment params based on JAX & scipy.optim
        Optimize u in the inner loop
        """
        x0, dims, shapes = self.dict_to_vec(x0)

        def _grad_x(x, umax, eps=1e-6):
            """ Gradient of x, given umax, for jit speed up"""
            umax = umax.reshape(-1, self.udim)
            xu = self.concate_xu(x, umax)
            _gxu = self.grad_xu(xu)
            _gx = _gxu[: -self.udim]
            _gu = jnp.expand_dims(_gxu[-self.udim :], 0)
            _hu = self.hessian_xu(xu)[-self.udim :, -self.udim :]
            _jux = self.jacobian_xu(xu)[-self.udim :, : -self.udim]
            gx = _gx + _gu.dot(-np.linalg.solve(_hu + eps, _jux))
            gx = gx.squeeze(0)
            if invert:
                gx = -1 * gx
            return gx

        def grad_x(x):
            umax, rmax, u_info = self.optimize_u(x, u0)
            gx = self.to_numpy(_grad_x(x, umax))
            return gx

        def cost_x(x):
            xu = self.concate_xu(x, u0)
            if invert:
                xu = -1 * xu
            return float(self.cost_xu(xu))

        # Optimize
        x0 = self.to_numpy(x0)
        xmax, rmax, info = fmin_l_bfgs_b(cost_x, x0, grad_x)
        x_info = {"dx": info["grad"]}
        xmax = self.vec_to_dict(xmax)
        return xmax, rmax, x_info
