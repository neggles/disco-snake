from diffusers import DPMSolverMultistepScheduler

dpmplusplus_2m_karras = DPMSolverMultistepScheduler(
    algorithm_type="dpmsolver++",
    solver_type="midpoint",
    prediction_type="epsilon",
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    thresholding=False,
)
