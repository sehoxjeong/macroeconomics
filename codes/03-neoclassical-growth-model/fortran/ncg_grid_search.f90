module neoclassical_growth

    implicit none
    real :: alpha, beta, delta, Α, kmin, kmax
    integer :: knum, max_iter, i
    real, dimension(:), allocatable :: kgrid, V, TV

    alpha = 1 / 3
    beta = 0.96
    delta = 1.0
    A = 1.0
    kmin = 1e-7


end module neoclassical_growth