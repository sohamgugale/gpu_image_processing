def is_cuda_available():
    try:
        import cupy
        cupy.zeros((1,))
        return True
    except Exception:
        return False
