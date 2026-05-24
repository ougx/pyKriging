!==============================================================================
! kriging_st_capi.f90
!
! ISO C Binding wrapper for the t_kriging_st Fortran module.
! Mirrors the structure of kriging_capi.f90; all entry points are prefixed
! with krige_st_ to avoid name collisions with the base library.
!
! All arrays are passed as explicit-shape dummies with a preceding size
! parameter so ctypes can pass raw pointers without hidden descriptors.
!
! New entry points vs. the base C API:
!   krige_st_set_obs          — adds time(nobs) and maxtlag
!   krige_st_set_grid         — adds time(ngrid)
!   krige_st_set_grid_block   — adds time arrays
!   krige_st_set_st_model     — sets global ST model parameters
!   krige_st_set_vgm_temporal — simplified temporal spec per (ivar,jvar)
!   krige_st_set_vgm_joint_sills — joint sills for sum-metric
!==============================================================================
module kriging_st_capi
  use iso_c_binding
  use kriging_st, only: t_kriging_st
  implicit none
  private

contains

  !=============================================================================
  ! Lifecycle
  !=============================================================================

  subroutine krige_st_create(handle) bind(C, name='krige_st_create')
    integer(c_intptr_t), intent(out) :: handle
    type(t_kriging_st), pointer :: obj
    allocate(obj)
    handle = transfer(c_loc(obj), handle)
  end subroutine krige_st_create

  subroutine krige_st_destroy(handle) bind(C, name='krige_st_destroy')
    integer(c_intptr_t), intent(inout) :: handle
    type(t_kriging_st), pointer :: obj
    call get_obj(handle, obj)
    call obj%finalize()
    deallocate(obj)
    handle = 0_c_intptr_t
  end subroutine krige_st_destroy

  !=============================================================================
  ! krige_st_initialize
  !=============================================================================
  subroutine krige_st_initialize(handle, &
      nvar, ndrift, unbias, nsim, &
      anisotropic_search, weight_correction, use_old_weight, &
      store_weight, cross_validation, write_mat, neglect_error, verbose, &
      weight_file, bounds, sk_mean, seed) &
      bind(C, name='krige_st_initialize')

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nvar, ndrift, unbias, nsim, seed
    integer(c_int),      intent(in), value :: anisotropic_search, weight_correction
    integer(c_int),      intent(in), value :: use_old_weight, store_weight
    integer(c_int),      intent(in), value :: cross_validation, write_mat, neglect_error, verbose
    character(kind=c_char), intent(in)     :: weight_file(*)
    real(c_double),      intent(in)        :: bounds(2)
    real(c_double),      intent(in), value :: sk_mean

    type(t_kriging_st), pointer :: obj
    call get_obj(handle, obj)

    call obj%initialize( &
      nvar               = int(nvar), &
      ndrift             = int(ndrift), &
      unbias             = int(unbias), &
      nsim               = int(nsim), &
      anisotropic_search = l(anisotropic_search), &
      weight_correction  = l(weight_correction), &
      use_old_weight     = l(use_old_weight), &
      store_weight       = l(store_weight), &
      cross_validation   = l(cross_validation), &
      write_mat          = l(write_mat), &
      neglect_error      = l(neglect_error), &
      verbose            = l(verbose), &
      weight_file        = c2fstr(weight_file), &
      bounds             = real(bounds), &
      sk_mean            = real(sk_mean), &
      seed               = int(seed))
  end subroutine krige_st_initialize

  !=============================================================================
  ! krige_st_set_st_model — global ST model parameters
  !   model:     0=sum_metric, 1=product_sum
  !   transform: 0=linear, 1=bounded, 2=power
  !   at:        joint temporal scale
  !   alpha:     power exponent (transform=2)
  !   k_ps:      product-sum k (model=1)
  !=============================================================================
  subroutine krige_st_set_st_model(handle, model, transform, at, alpha, k_ps) &
      bind(C, name='krige_st_set_st_model')
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: model, transform
    real(c_double),      intent(in), value :: at, alpha, k_ps
    type(t_kriging_st), pointer :: obj
    call get_obj(handle, obj)
    call obj%set_st_model(int(model), int(transform), real(at), real(alpha), real(k_ps))
  end subroutine krige_st_set_st_model

  !=============================================================================
  ! krige_st_set_obs
  !   time    : [nobs]   observation times
  !   maxtlag : maximum temporal search lag (physical units; pass huge for no limit)
  !=============================================================================
  subroutine krige_st_set_obs(handle, ivar, nobs, &
      coord, value, time, variance, nmax, maxdist, maxtlag) &
      bind(C, name='krige_st_set_obs')
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, nobs, nmax
    real(c_double),      intent(in)        :: coord(3, nobs)
    real(c_double),      intent(in)        :: value(nobs)
    real(c_double),      intent(in)        :: time(nobs)
    real(c_double),      intent(in)        :: variance(nobs)
    real(c_double),      intent(in), value :: maxdist, maxtlag
    type(t_kriging_st), pointer :: obj
    call get_obj(handle, obj)
    call obj%set_obs(int(ivar), real(coord), real(value), real(time), &
      variance = real(variance), &
      nmax     = int(nmax), &
      maxdist  = real(maxdist), &
      maxtlag  = real(maxtlag))
  end subroutine krige_st_set_obs

  !=============================================================================
  ! krige_st_set_obs_drift
  !=============================================================================
  subroutine krige_st_set_obs_drift(handle, ivar, ndrift_c, nobs, drift) &
      bind(C, name='krige_st_set_obs_drift')
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, ndrift_c, nobs
    real(c_double),      intent(in)        :: drift(ndrift_c, nobs)
    type(t_kriging_st), pointer :: obj
    call get_obj(handle, obj)
    call obj%set_obs_drift(int(ivar), real(drift))
  end subroutine krige_st_set_obs_drift

  !=============================================================================
  ! krige_st_set_vgm — add one spatial nested structure to vgm(ivar,jvar)%cs
  !=============================================================================
  subroutine krige_st_set_vgm(handle, ivar, jvar, spec) &
      bind(C, name='krige_st_set_vgm')
    integer(c_intptr_t),    intent(in), value :: handle
    integer(c_int),         intent(in), value :: ivar, jvar
    character(kind=c_char), intent(in)        :: spec(*)
    type(t_kriging_st), pointer :: obj
    call get_obj(handle, obj)
    call obj%set_vgm(int(ivar), int(jvar), c2fstr(spec))
  end subroutine krige_st_set_vgm

  !=============================================================================
  ! krige_st_set_vgm_temporal — add one temporal nested structure
  !   spec: null-terminated "vtype nugget sill at_k"
  !=============================================================================
  subroutine krige_st_set_vgm_temporal(handle, ivar, jvar, spec) &
      bind(C, name='krige_st_set_vgm_temporal')
    integer(c_intptr_t),    intent(in), value :: handle
    integer(c_int),         intent(in), value :: ivar, jvar
    character(kind=c_char), intent(in)        :: spec(*)
    type(t_kriging_st), pointer :: obj
    call get_obj(handle, obj)
    call obj%set_vgm_temporal(int(ivar), int(jvar), c2fstr(spec))
  end subroutine krige_st_set_vgm_temporal

  !=============================================================================
  ! krige_st_set_vgm_joint_sills — joint sills for sum-metric model
  !   sills : [n]  one per spatial nested structure of cs
  !=============================================================================
  subroutine krige_st_set_vgm_joint_sills(handle, ivar, jvar, n, sills) &
      bind(C, name='krige_st_set_vgm_joint_sills')
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, jvar, n
    real(c_double),      intent(in)        :: sills(n)
    type(t_kriging_st), pointer :: obj
    call get_obj(handle, obj)
    call obj%set_vgm_joint_sills(int(ivar), int(jvar), real(sills), int(n))
  end subroutine krige_st_set_vgm_joint_sills

  !=============================================================================
  ! krige_st_set_grid — point estimation targets with times
  !=============================================================================
  subroutine krige_st_set_grid(handle, ngrid, coord, time, rangescale, localnugget) &
      bind(C, name='krige_st_set_grid')
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ngrid
    real(c_double),      intent(in)        :: coord(3, ngrid)
    real(c_double),      intent(in)        :: time(ngrid)
    real(c_double),      intent(in)        :: rangescale(ngrid)
    real(c_double),      intent(in)        :: localnugget(ngrid)
    type(t_kriging_st), pointer :: obj
    call get_obj(handle, obj)
    call obj%set_grid(real(coord), real(time), real(rangescale), real(localnugget))
  end subroutine krige_st_set_grid

  !=============================================================================
  ! krige_st_set_grid_block — block estimation targets with integration nodes
  !=============================================================================
  subroutine krige_st_set_grid_block(handle, nblocks, coord, time, &
      nblockpnt, npnts_total, blockcoord, blocktime, pointweight, &
      rangescale, localnugget) &
      bind(C, name='krige_st_set_grid_block')
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nblocks, npnts_total
    real(c_double),      intent(in)        :: coord(3, nblocks)
    real(c_double),      intent(in)        :: time(nblocks)
    integer(c_int),      intent(in)        :: nblockpnt(nblocks)
    real(c_double),      intent(in)        :: blockcoord(3, npnts_total)
    real(c_double),      intent(in)        :: blocktime(npnts_total)
    real(c_double),      intent(in)        :: pointweight(npnts_total)
    real(c_double),      intent(in)        :: rangescale(nblocks)
    real(c_double),      intent(in)        :: localnugget(nblocks)
    type(t_kriging_st), pointer :: obj
    call get_obj(handle, obj)
    call obj%set_grid_block( &
      real(coord), real(time), int(nblockpnt), &
      real(blockcoord), real(blocktime), real(pointweight), &
      real(rangescale), real(localnugget))
  end subroutine krige_st_set_grid_block

  !=============================================================================
  ! krige_st_set_grid_cv — cross-validation mode
  !=============================================================================
  subroutine krige_st_set_grid_cv(handle) bind(C, name='krige_st_set_grid_cv')
    integer(c_intptr_t), intent(in), value :: handle
    type(t_kriging_st), pointer :: obj
    call get_obj(handle, obj)
    call obj%set_grid_cv()
  end subroutine krige_st_set_grid_cv

  !=============================================================================
  ! krige_st_set_grid_drift
  !=============================================================================
  subroutine krige_st_set_grid_drift(handle, ndrift_c, nblocks, drift) &
      bind(C, name='krige_st_set_grid_drift')
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ndrift_c, nblocks
    real(c_double),      intent(in)        :: drift(ndrift_c, nblocks)
    type(t_kriging_st), pointer :: obj
    call get_obj(handle, obj)
    call obj%set_grid_drift(real(drift))
  end subroutine krige_st_set_grid_drift

  !=============================================================================
  ! krige_st_set_sim — SGSIM random path and samples
  !=============================================================================
  subroutine krige_st_set_sim(handle, nblocks, randpath, nsim_c, sample) &
      bind(C, name='krige_st_set_sim')
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nblocks, nsim_c
    integer(c_int),      intent(in)        :: randpath(nblocks)
    real(c_double),      intent(in)        :: sample(nsim_c, nblocks)
    type(t_kriging_st), pointer :: obj
    call get_obj(handle, obj)
    call obj%set_sim(randpath=int(randpath), sample=real(sample))
  end subroutine krige_st_set_sim

  !=============================================================================
  ! krige_st_set_search — build spatial KD-tree for variable ivar
  !=============================================================================
  subroutine krige_st_set_search(handle, ivar, anis1, anis2, azimuth, dip, plunge) &
      bind(C, name='krige_st_set_search')
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar
    real(c_double),      intent(in), value :: anis1, anis2, azimuth, dip, plunge
    type(t_kriging_st), pointer :: obj
    call get_obj(handle, obj)
    call obj%set_search(int(ivar), &
      anis1=real(anis1), anis2=real(anis2), &
      azimuth=real(azimuth), dip=real(dip), plunge=real(plunge))
  end subroutine krige_st_set_search

  !=============================================================================
  ! krige_st_solve
  !=============================================================================
  subroutine krige_st_solve(handle) bind(C, name='krige_st_solve')
    integer(c_intptr_t), intent(in), value :: handle
    type(t_kriging_st), pointer :: obj
    call get_obj(handle, obj)
    call obj%solve()
  end subroutine krige_st_solve

  !=============================================================================
  ! Result getters (same pattern as base)
  !=============================================================================

  subroutine krige_st_get_nblocks(handle, n) bind(C, name='krige_st_get_nblocks')
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(out)       :: n
    type(t_kriging_st), pointer :: obj
    call get_obj(handle, obj)
    n = int(obj%block%n, c_int)
  end subroutine krige_st_get_nblocks

  subroutine krige_st_get_nsim(handle, n) bind(C, name='krige_st_get_nsim')
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(out)       :: n
    type(t_kriging_st), pointer :: obj
    call get_obj(handle, obj)
    n = max(int(obj%nsim, c_int), 1_c_int)
  end subroutine krige_st_get_nsim

  subroutine krige_st_get_estimate(handle, nsim_c, nblocks, out) &
      bind(C, name='krige_st_get_estimate')
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nsim_c, nblocks
    real(c_double),      intent(out)       :: out(nsim_c, nblocks)
    type(t_kriging_st), pointer :: obj
    call get_obj(handle, obj)
    out = real(obj%block%estimate(1:nsim_c, 1:nblocks), c_double)
  end subroutine krige_st_get_estimate

  subroutine krige_st_get_variance(handle, nblocks, out) &
      bind(C, name='krige_st_get_variance')
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nblocks
    real(c_double),      intent(out)       :: out(nblocks)
    type(t_kriging_st), pointer :: obj
    call get_obj(handle, obj)
    out = real(obj%block%variance(1:nblocks), c_double)
  end subroutine krige_st_get_variance

  !=============================================================================
  ! Internal helpers
  !=============================================================================

  subroutine get_obj(handle, obj)
    integer(c_intptr_t), intent(in), value :: handle
    type(t_kriging_st),  pointer           :: obj
    type(c_ptr) :: cptr
    cptr = transfer(handle, cptr)
    call c_f_pointer(cptr, obj)
  end subroutine get_obj

  function c2fstr(cstr) result(fstr)
    character(kind=c_char), intent(in) :: cstr(*)
    character(len=1024) :: fstr
    integer :: i
    fstr = ''
    do i = 1, 1024
      if (cstr(i) == c_null_char) exit
      fstr(i:i) = cstr(i)
    end do
  end function c2fstr

  elemental function l(v) result(r)
    integer(c_int), intent(in), value :: v
    logical :: r
    r = (v == 1_c_int)
  end function l

end module kriging_st_capi
