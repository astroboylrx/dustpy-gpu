API Extensions
==============

DustPy-GPU preserves the upstream ``dustpy`` API. The main public extension is
backend selection when constructing a simulation.

.. py:class:: dustpy.Simulation(backend=None, **kwargs)

   Create a DustPy simulation using ``numpy``, ``cupy``, ``torch``, or ``auto``.
   ``None`` selects NumPy and preserves upstream behavior.

   .. py:property:: backend

      The resolved backend name for this simulation.

   .. py:method:: run(*, RL_debug=False)

      Run the configured integrator. Set ``RL_debug=True`` to print the optional
      DustPy-GPU runtime diagnostics::

         sim.run(RL_debug=True)

      The first accepted cycle and every ``sim.RL_ncycle_out`` cycles thereafter
      (100 by default) print a line such as::

         [RL_debug]: cycle=      100, t=    1000.000yr, dt=1.0000e+01/9.8000e+00/1.0000e+01yr, n_dt↘=     2, dM_M⊕=1.2000e-08/3.0000e-10, wt=4.2500e+00

      ``cycle``
         Number of accepted integration steps since ``run()`` started.

      ``t``
         Current simulation time in years.

      ``dt``
         Current, rolling-mean, and rolling-median timestep in years. The rolling
         values use the latest 100 accepted steps and become representative after
         the initial buffer has filled.

      ``n_dt↘``
         Cumulative count of implicit dust steps retried with a smaller timestep
         because the candidate solution required excessive floor mass injection.

      ``dM_M⊕``
         Cumulative floor mass added, followed by cumulative sanitizer mass
         removed, both in Earth masses.

      ``wt``
         Summed wall time in seconds for the latest 100 accepted-step intervals.

For the standard model fields and customization API, use the `upstream DustPy
module reference <https://stammler.github.io/dustpy/api.html>`_. Backend array
helpers are provided by `Simframe-GPU
<https://github.com/astroboylrx/simframe-gpu>`_.
