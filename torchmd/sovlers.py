from torchmd.tinydiffeq import _check_inputs, _flatten, _flatten_convert_none_to_zeros
from torchmd.tinydiffeq import RK4, FixedGridODESolver

import torch 
from torch import nn

'''
    I need to think how to write generatic verlet update for both forward and adjoint integration 
'''

class NHVerlet(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        return NHverlet_update(func, t, dt, y)

class Verlet(FixedGridODESolver):

    def step_func(self, func, t, dt, y):
        return verlet_update(func, t, dt, y)

def verlet_update(func, t, dt, y):

    NUM_VAR = 2

    if len(y) == NUM_VAR: # integrator in the forward call 
        a_0, v_0 = func(t, y)

        # update half step 
        v_step_half = 0.5 *  a_0 * dt 

        # update full step in positions 
        q_step_full = (y[0] + v_step_half) * dt 

        # gradient full at t + dt 
        a_dt, v_half = func(t, (y[0] + v_step_half, y[1] + q_step_full))

        # full step update 
        v_step_full = v_step_half + 0.5 * a_dt * dt

        return tuple((v_step_full, q_step_full))
    
    elif len(y) == NUM_VAR * 2 + 2: # integrator in the backward call

        v_full, x_full, vad_full, xad_full = y[0], y[1], y[2], y[3]

        dv, dx, vad_vjp_full, xad_vjp_full, vjp_t, vjp_params= func(t, y)  # compute dy, and vjps 

        # Reverse integrator 
        v_step_half = 1/2 * dv * dt 
        v_half = v_full - v_step_half
        x_step_full = v_half * dt 
        x_0 = x_full - x_step_full

        #print(vad_vjp, xad_vjp)

        # So vad_vjp = xad_full?

        # func is the automatically generated ODE for adjoints 
        # dydt_0 variable name is a bit confusing(it even confused me after 3 months of writing this snippit),
        # I need to change to the right adjoint definition -> dLdv, dLdq or v_hat and q_t  

        # more importantly are there better way to integrate the adjoint state other than midpoint integration 

        #vadjoint_step_half = 1/2 * dydt_0[0 + 3] * dt # update adjoint state 
        
        # func returns the infiniesmal changes of different states 

        #xad_full_tmp = xad_vjp
        #vad_full = vad_full

        dxad_full = xad_vjp_full * dt * 0.5
        dvad_half = (xad_full + dxad_full) * dt #* 0.5 # alternatively dvad_half = dvad_half = xad_full *  dt

        vad_half = vad_full + dvad_half 

        #xad_full = xad_vjp
        #vad_full = vad_vjp_full

        #print(vad_full, xad_full, vad_vjp, xad_vjp, vad_half)

        dLdt_half = vjp_t  * dt 
        dLdpar_half = vjp_params * 0.5 * dt # par_adjoint 

        #xad_vjp_half = xad_vjp * dt * 0.5
        
        dv, dx, vad_vjp_half, xad_vjp_half, vjp_t, vjp_params = func(t, (v_half, x_0, 
                    vad_half, xad_full + dxad_full , 
                    y[4] + dLdt_half, y[5] + dLdpar_half
                   ))

        v_step_full = v_step_half - dv * dt * 0.5 

        dvad_0 = vad_vjp_full * dt # update adjoint state 
        dxad_0 = xad_vjp_half * dt * 0.5#   xad_vjp_half * dt #+  xad_vjp_full * dt * 0.5

        dLdt_step = vjp_t * dt 
        dLdpar_step = vjp_params * dt * 0.5
        
        return (v_step_full, x_step_full,
                (dvad_half), (dxad_0 + dxad_full), 
                dLdt_step,  dLdpar_half * 2)
    else:
        raise ValueError("received {} argumets integration, but should be {} for the forward call or {} for the backward call".format(
                len(y), NUM_VAR, 2 * NUM_VAR + 2))

def NHverlet_update(func, t, dt, y):

    NUM_VAR = 3

    if len(y) == NUM_VAR: # integrator in the forward call 
        a_0, v_0, dpvdt_0 = func(t, y)

        # update half step 
        v_step_half = 1/2 *  a_0 * dt 
        pv_step_half = 1/2 * dpvdt_0 * dt

        # update full step in positions 
        q_step_full = (y[0] + v_step_half) * dt 

        # gradient full at t + dt 
        a_dt, v_half, dpvdt_half = func(t, (y[0] + v_step_half, y[1] + q_step_full, y[2] + pv_step_half))

        # full step update 
        v_step_full = v_step_half + 1/2 * a_dt * dt
        pv_step_full = pv_step_half + 1/2 * dpvdt_half * dt 

        return tuple((v_step_full, q_step_full, pv_step_full))
    
    elif len(y) == NUM_VAR * 2 + 2: # integrator in the backward call 
        dydt_0 = func(t, y)
        
        v_step_half = 1/2 * dydt_0[0] * dt 
        #vadjoint_step_half = 1/2 * dydt_0[0 + 3] * dt # update adjoint state 
        
        pv_step_half = 1/2 * dydt_0[2] * dt 
        #pvadjoint_step_half = 1/2 * dydt_0[2 + 3] * dt 
        
        q_step_full = (y[0] + v_step_half) * dt 
        
        # half step adjoint update 
        vadjoint_half = dydt_0[3] * 0.5 * dt # update adjoint state 
        qadjoint_half = dydt_0[4] * 0.5 * dt 
        pvadjoint_half = dydt_0[5] * 0.5 * dt
        dLdt_half = dydt_0[6] * 0.5 * dt 
        dLdpar_half = dydt_0[7] * 0.5 * dt 
        
        dydt_mid = func(t, (y[0] + v_step_half, y[1] + q_step_full, y[2] + pv_step_half, 
                    y[3] + vadjoint_half, y[4] + qadjoint_half, y[5] + pvadjoint_half, 
                    y[6] + dLdt_half, y[7] + dLdpar_half
                   ))

        v_step_full = v_step_half + 1/2 * dydt_mid[0] * dt 
        pv_step_full = pv_step_half + 1/2 * dydt_mid[2] * dt 
        
        # half step adjoint update 
        vadjoint_step = dydt_mid[3] * dt # update adjoint state 
        qadjoint_step = dydt_mid[4] * dt 
        pvadjoint_step = dydt_mid[5] * dt
        dLdt_step = dydt_mid[6] * dt 
        dLdpar_step = dydt_mid[7] * dt         
        
        return (v_step_full, q_step_full, pv_step_full, 
                vadjoint_step, qadjoint_step, pvadjoint_step,
                dLdt_step, dLdpar_step)

    else:
        raise ValueError("received {} argumets integration, but should be {} for the forward call or {} for the backward call".format(
                len(y), NUM_VAR, 2 * NUM_VAR + 2))


def odeint(func, y0, t, rtol=1e-7, atol=1e-9, method=None, options=None):

    SOLVERS = {
    'rk4': RK4,
    'NH_verlet': NHVerlet,
    'verlet': Verlet
    }

    tensor_input, func, y0, t = _check_inputs(func, y0, t)

    if options is None:
        options = {}
    elif method is None:
        raise ValueError('cannot supply `options` without specifying `method`')

    if method is None:
        method = 'dopri5'

    solver = SOLVERS[method](func, y0, rtol=rtol, atol=atol, **options)
    solution = solver.integrate(t)
    if tensor_input:
        solution = solution[0]
    return solution


class OdeintAdjointMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        assert len(args) >= 8, 'Internal error: all arguments required.'
        y0, func, t, flat_params, rtol, atol, method, options = \
            args[:-7], args[-7], args[-6], args[-5], args[-4], args[-3], args[-2], args[-1]

        ctx.func, ctx.rtol, ctx.atol, ctx.method, ctx.options = func, rtol, atol, method, options

        with torch.no_grad():
            ans = odeint(func, y0, t, rtol=rtol, atol=atol, method=method, options=options)
        ctx.save_for_backward(t, flat_params, *ans)
        return ans

    @staticmethod
    def backward(ctx, *grad_output):

        t, flat_params, *ans = ctx.saved_tensors
        ans = tuple(ans)
        func, rtol, atol, method, options = ctx.func, ctx.rtol, ctx.atol, ctx.method, ctx.options
        n_tensors = len(ans)
        f_params = tuple(func.parameters())

        # TODO: use a nn.Module and call odeint_adjoint to implement higher order derivatives.
        def augmented_dynamics(t, y_aug):
            # Dynamics of the original system augmented with
            # the adjoint wrt y, and an integrator wrt t and args.
            y, adj_y = y_aug[:n_tensors], y_aug[n_tensors:2 * n_tensors]  # Ignore adj_time and adj_params.

            with torch.set_grad_enabled(True):
                t = t.to(y[0].device).detach().requires_grad_(True)
                y = tuple(y_.detach().requires_grad_(True) for y_ in y) # get state variables 
                func_eval = func(t, y)
                vjp_t, *vjp_y_and_params = torch.autograd.grad(
                    func_eval, (t,) + y + f_params,
                    tuple(-adj_y_ for adj_y_ in adj_y), allow_unused=True, retain_graph=True
                ) 

            vjp_y = vjp_y_and_params[:n_tensors]
            vjp_params = vjp_y_and_params[n_tensors:]

            # autograd.grad returns None if no gradient, set to zero.
            vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
            vjp_y = tuple(torch.zeros_like(y_) if vjp_y_ is None else vjp_y_ for vjp_y_, y_ in zip(vjp_y, y))
            vjp_params = _flatten_convert_none_to_zeros(vjp_params, f_params)

            if len(f_params) == 0:
                vjp_params = torch.tensor(0.).to(vjp_y[0])
            return (*func_eval, *vjp_y, vjp_t, vjp_params)

        T = ans[0].shape[0]
        with torch.no_grad():
            adj_y = tuple(grad_output_[-1] for grad_output_ in grad_output)
            adj_params = torch.zeros_like(flat_params)
            adj_time = torch.tensor(0.).to(t)
            time_vjps = []
            for i in range(T - 1, 0, -1):

                ans_i = tuple(ans_[i] for ans_ in ans)

                grad_output_i = tuple(grad_output_[i] for grad_output_ in grad_output)
                func_i = func(t[i], ans_i)

                # Compute the effect of moving the current time measurement point.
                dLd_cur_t = sum(
                    torch.dot(func_i_.reshape(-1), grad_output_i_.reshape(-1)).reshape(1)
                    for func_i_, grad_output_i_ in zip(func_i, grad_output_i)
                )
                adj_time = adj_time - dLd_cur_t
                time_vjps.append(dLd_cur_t)

                # Run the augmented system backwards in time.
                if adj_params.numel() == 0:
                    adj_params = torch.tensor(0.).to(adj_y[0])
                aug_y0 = (*ans_i, *adj_y, adj_time, adj_params)
                aug_ans = odeint(
                    augmented_dynamics, aug_y0,
                    torch.tensor([t[i], t[i - 1]]), rtol=rtol, atol=atol, method=method, options=options
                )

                # Unpack aug_ans.
                adj_y = aug_ans[n_tensors:2 * n_tensors]
                adj_time = aug_ans[2 * n_tensors]
                adj_params = aug_ans[2 * n_tensors + 1]

                adj_y = tuple(adj_y_[1] if len(adj_y_) > 0 else adj_y_ for adj_y_ in adj_y)
                if len(adj_time) > 0: adj_time = adj_time[1]
                if len(adj_params) > 0: adj_params = adj_params[1]

                adj_y = tuple(adj_y_ + grad_output_[i - 1] for adj_y_, grad_output_ in zip(adj_y, grad_output))

                del aug_y0, aug_ans

            time_vjps.append(adj_time)
            time_vjps = torch.cat(time_vjps[::-1])

            return (*adj_y, None, time_vjps, adj_params, None, None, None, None, None)


def odeint_adjoint(func, y0, t, rtol=1e-6, atol=1e-12, method=None, options=None):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if not isinstance(func, nn.Module):
        raise ValueError('func is required to be an instance of nn.Module.')

    tensor_input = False
    if torch.is_tensor(y0):

        class TupleFunc(nn.Module):

            def __init__(self, base_func):
                super(TupleFunc, self).__init__()
                self.base_func = base_func

            def forward(self, t, y):
                return (self.base_func(t, y[0]),)

        tensor_input = True
        y0 = (y0,)
        func = TupleFunc(func)

    flat_params = _flatten(func.parameters())
    ys = OdeintAdjointMethod.apply(*y0, func, t, flat_params, rtol, atol, method, options)

    if tensor_input:
        ys = ys[0]
    return ys