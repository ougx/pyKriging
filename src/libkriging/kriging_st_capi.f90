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
  use kriging_err, only: kriging_clear_error, kriging_ierr, kriging_error, kriging_failed
  implicit none
  private

  ! Same registry-handle pattern as kriging_capi: ctypes sees an integer slot,
  ! while Fortran keeps ownership of the non-C-interoperable derived type.
  type kriging_st_handle_slot
    type(t_kriging_st), pointer :: obj => null()
  end type kriging_st_handle_slot

  type(kriging_st_handle_slot), allocatable, save :: kriging_st_registry(:)

contains

  !=============================================================================
  ! Lifecycle
  !=============================================================================

  integer(c_int) function krige_st_create(handle) bind(C, name='krige_st_create') result(ierr)
    integer(c_intptr_t), intent(out) :: handle
    type(t_kriging_st), pointer :: obj
    integer :: stat
    call kriging_clear_error()
    handle = 0_c_intptr_t
    allocate(obj, stat=stat)
    if (stat /= 0) then
      call kriging_error('krige_st_create', 'Failed to allocate t_kriging_st object')
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call store_obj(obj, handle)
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_create

  integer(c_int) function krige_st_destroy(handle) bind(C, name='krige_st_destroy') result(ierr)
    integer(c_intptr_t), intent(inout) :: handle
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%finalize()
    deallocate(obj)
    call release_obj(handle)
    handle = 0_c_intptr_t
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_destroy

  !=============================================================================
  ! krige_st_initialize
  !=============================================================================
  integer(c_int) function krige_st_initialize(handle, &
      nvar, ndrift, unbias, nsim, &
      anisotropic_search, weight_correction, use_old_weight, &
      store_weight, cross_validation, write_mat, neglect_error, verbose, &
      weight_file, bounds, sk_mean, seed) &
      bind(C, name='krige_st_initialize') result(ierr)

    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nvar, ndrift, unbias, nsim, seed
    integer(c_int),      intent(in), value :: anisotropic_search, weight_correction
    integer(c_int),      intent(in), value :: use_old_weight, store_weight
    integer(c_int),      intent(in), value :: cross_validation, write_mat, neglect_error, verbose
    character(kind=c_char), intent(in)     :: weight_file(*)
    real(c_double),      intent(in)        :: bounds(2)
    real(c_double),      intent(in), value :: sk_mean

    type(t_kriging_st), pointer :: obj
    real :: fbounds(2)
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    fbounds = real(bounds)

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
      bounds             = fbounds, &
      sk_mean            = real(sk_mean), &
      seed               = int(seed))
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_initialize

  !=============================================================================
  ! krige_st_set_st_model — global ST model parameters
  !   model:     0=sum_metric, 1=product_sum
  !   transform: 0=linear, 1=bounded, 2=power
  !   at:        joint temporal scale
  !   alpha:     power exponent (transform=2)
  !   k_ps:      product-sum k (model=1)
  !=============================================================================
  integer(c_int) function krige_st_set_st_model(handle, model, transform, at, alpha, k_ps) &
      bind(C, name='krige_st_set_st_model') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: model, transform
    real(c_double),      intent(in), value :: at, alpha, k_ps
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_st_model(int(model), int(transform), real(at), real(alpha), real(k_ps))
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_st_model

  !=============================================================================
  ! krige_st_set_obs
  !   time    : [nobs]   observation times
  !   maxtlag : maximum temporal search lag (physical units; pass huge for no limit)
  !=============================================================================
  integer(c_int) function krige_st_set_obs(handle, ivar, nobs, &
      coord, value, time, variance, nmax, maxdist, maxtlag) &
      bind(C, name='krige_st_set_obs') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, nobs, nmax
    real(c_double),      intent(in)        :: coord(3, nobs)
    real(c_double),      intent(in)        :: value(nobs)
    real(c_double),      intent(in)        :: time(nobs)
    real(c_double),      intent(in)        :: variance(nobs)
    real(c_double),      intent(in), value :: maxdist, maxtlag
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_obs(int(ivar), real(coord), real(value), real(time), &
      variance = real(variance), &
      nmax     = int(nmax), &
      maxdist  = real(maxdist), &
      maxtlag  = real(maxtlag))
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_obs

  !=============================================================================
  ! krige_st_set_obs_drift
  !=============================================================================
  integer(c_int) function krige_st_set_obs_drift(handle, ivar, ndrift_c, nobs, drift) &
      bind(C, name='krige_st_set_obs_drift') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, ndrift_c, nobs
    real(c_double),      intent(in)        :: drift(ndrift_c, nobs)
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_obs_drift(int(ivar), real(drift))
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_obs_drift

  !=============================================================================
  ! krige_st_set_vgm — add one spatial nested structure to vgm(ivar,jvar)%cs
  !=============================================================================
  integer(c_int) function krige_st_set_vgm(handle, ivar, jvar, spec) &
      bind(C, name='krige_st_set_vgm') result(ierr)
    integer(c_intptr_t),    intent(in), value :: handle
    integer(c_int),         intent(in), value :: ivar, jvar
    character(kind=c_char), intent(in)        :: spec(*)
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_vgm(int(ivar), int(jvar), c2fstr(spec))
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_vgm

  !=============================================================================
  ! krige_st_set_vgm_temporal — add one temporal nested structure
  !   spec: null-terminated "vtype nugget sill at_k"
  !=============================================================================
  integer(c_int) function krige_st_set_vgm_temporal(handle, ivar, jvar, spec) &
      bind(C, name='krige_st_set_vgm_temporal') result(ierr)
    integer(c_intptr_t),    intent(in), value :: handle
    integer(c_int),         intent(in), value :: ivar, jvar
    character(kind=c_char), intent(in)        :: spec(*)
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_vgm_temporal(int(ivar), int(jvar), c2fstr(spec))
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_vgm_temporal

  !=============================================================================
  ! krige_st_set_vgm_joint_sills — joint sills for sum-metric model
  !   sills : [n]  one per spatial nested structure of cs
  !=============================================================================
  integer(c_int) function krige_st_set_vgm_joint_sills(handle, ivar, jvar, n, sills) &
      bind(C, name='krige_st_set_vgm_joint_sills') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar, jvar, n
    real(c_double),      intent(in)        :: sills(n)
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_vgm_joint_sills(int(ivar), int(jvar), real(sills), int(n))
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_vgm_joint_sills

  !=============================================================================
  ! krige_st_set_grid — point estimation targets with times
  !=============================================================================
  integer(c_int) function krige_st_set_grid(handle, ngrid, coord, time, rangescale, localnugget) &
      bind(C, name='krige_st_set_grid') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ngrid
    real(c_double),      intent(in)        :: coord(3, ngrid)
    real(c_double),      intent(in)        :: time(ngrid)
    real(c_double),      intent(in)        :: rangescale(ngrid)
    real(c_double),      intent(in)        :: localnugget(ngrid)
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_grid(real(coord), real(time), real(rangescale), real(localnugget))
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_grid

  !=============================================================================
  ! krige_st_set_grid_block — block estimation targets with integration nodes
  !=============================================================================
  integer(c_int) function krige_st_set_grid_block(handle, nblocks, coord, time, &
      nblockpnt, npnts_total, blockcoord, blocktime, pointweight, &
      rangescale, localnugget) &
      bind(C, name='krige_st_set_grid_block') result(ierr)
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
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_grid_block( &
      real(coord), real(time), int(nblockpnt), &
      real(blockcoord), real(blocktime), real(pointweight), &
      real(rangescale), real(localnugget))
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_grid_block

  !=============================================================================
  ! krige_st_set_grid_cv — cross-validation mode
  !=============================================================================
  integer(c_int) function krige_st_set_grid_cv(handle) bind(C, name='krige_st_set_grid_cv') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_grid_cv()
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_grid_cv

  !=============================================================================
  ! krige_st_set_grid_drift
  !=============================================================================
  integer(c_int) function krige_st_set_grid_drift(handle, ndrift_c, nblocks, drift) &
      bind(C, name='krige_st_set_grid_drift') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ndrift_c, nblocks
    real(c_double),      intent(in)        :: drift(ndrift_c, nblocks)
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_grid_drift(real(drift))
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_grid_drift

  !=============================================================================
  ! krige_st_set_sim — SGSIM random path and samples
  !=============================================================================
  integer(c_int) function krige_st_set_sim(handle, nblocks, randpath, nsim_c, sample) &
      bind(C, name='krige_st_set_sim') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nblocks, nsim_c
    integer(c_int),      intent(in)        :: randpath(nblocks)
    real(c_double),      intent(in)        :: sample(nsim_c, nblocks)
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_sim(randpath=int(randpath), sample=real(sample))
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_sim

  !=============================================================================
  ! krige_st_set_search — build spatial KD-tree for variable ivar
  !=============================================================================
  integer(c_int) function krige_st_set_search(handle, ivar, anis1, anis2, azimuth, dip, plunge) &
      bind(C, name='krige_st_set_search') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: ivar
    real(c_double),      intent(in), value :: anis1, anis2, azimuth, dip, plunge
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%set_search(int(ivar), &
      anis1=real(anis1), anis2=real(anis2), &
      azimuth=real(azimuth), dip=real(dip), plunge=real(plunge))
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_set_search

  !=============================================================================
  ! krige_st_solve
  !=============================================================================
  integer(c_int) function krige_st_solve(handle) bind(C, name='krige_st_solve') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    call obj%solve()
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_solve

  !=============================================================================
  ! Result getters (same pattern as base)
  !=============================================================================

  integer(c_int) function krige_st_get_nblocks(handle, n) bind(C, name='krige_st_get_nblocks') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(out)       :: n
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    n = 0_c_int
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    n = int(obj%block%n, c_int)
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_get_nblocks

  integer(c_int) function krige_st_get_nsim(handle, n) bind(C, name='krige_st_get_nsim') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(out)       :: n
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    n = 0_c_int
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    n = max(int(obj%nsim, c_int), 1_c_int)
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_get_nsim

  integer(c_int) function krige_st_get_estimate(handle, nsim_c, nblocks, out) &
      bind(C, name='krige_st_get_estimate') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nsim_c, nblocks
    real(c_double),      intent(out)       :: out(nsim_c, nblocks)
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    out = real(obj%block%estimate(1:nsim_c, 1:nblocks), c_double)
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_get_estimate

  integer(c_int) function krige_st_get_variance(handle, nblocks, out) &
      bind(C, name='krige_st_get_variance') result(ierr)
    integer(c_intptr_t), intent(in), value :: handle
    integer(c_int),      intent(in), value :: nblocks
    real(c_double),      intent(out)       :: out(nblocks)
    type(t_kriging_st), pointer :: obj
    call kriging_clear_error()
    call get_obj(handle, obj)
    if (kriging_failed()) then
      ierr = int(kriging_ierr(), c_int)
      return
    end if
    out = real(obj%block%variance(1:nblocks), c_double)
    ierr = int(kriging_ierr(), c_int)
  end function krige_st_get_variance

  !=============================================================================
  ! Internal helpers
  !=============================================================================

  subroutine get_obj(handle, obj)
    ! Resolve the slot index back to the live ST object.  Bad handles become an
    ! ierr/error-message pair instead of an access violation.
    integer(c_intptr_t), intent(in), value :: handle
    type(t_kriging_st),  pointer           :: obj
    integer :: idx
    nullify(obj)
    if (handle == 0_c_intptr_t) then
      call kriging_error('kriging_st_capi', 'Null kriging object handle')
      return
    end if
    idx = int(handle)
    if (.not. allocated(kriging_st_registry) .or. idx < 1 .or. idx > size(kriging_st_registry)) then
      call kriging_error('kriging_st_capi', 'Invalid kriging object handle')
      return
    end if
    if (associated(kriging_st_registry(idx)%obj)) obj => kriging_st_registry(idx)%obj
    if (.not. associated(obj)) &
      call kriging_error('kriging_st_capi', 'Invalid kriging object handle')
  end subroutine get_obj

  subroutine store_obj(obj, handle)
    ! Allocate a stable slot for this object; existing handles remain valid when
    ! the registry grows.
    type(t_kriging_st), pointer, intent(in) :: obj
    integer(c_intptr_t), intent(out) :: handle
    integer :: i

    if (.not. allocated(kriging_st_registry)) allocate(kriging_st_registry(16))

    do i = 1, size(kriging_st_registry)
      if (.not. associated(kriging_st_registry(i)%obj)) then
        kriging_st_registry(i)%obj => obj
        handle = int(i, c_intptr_t)
        return
      end if
    end do

    call grow_registry()
    do i = 1, size(kriging_st_registry)
      if (.not. associated(kriging_st_registry(i)%obj)) then
        kriging_st_registry(i)%obj => obj
        handle = int(i, c_intptr_t)
        return
      end if
    end do

    handle = 0_c_intptr_t
    call kriging_error('krige_st_create', 'Failed to allocate a kriging ST handle slot')
  end subroutine store_obj

  subroutine release_obj(handle)
    ! Make the slot available for reuse after destroy without renumbering any
    ! other live handles.
    integer(c_intptr_t), intent(in), value :: handle
    integer :: idx
    idx = int(handle)
    if (allocated(kriging_st_registry) .and. idx >= 1 .and. idx <= size(kriging_st_registry)) &
      nullify(kriging_st_registry(idx)%obj)
  end subroutine release_obj

  subroutine grow_registry()
    ! Expand the saved registry while preserving pointer associations.
    type(kriging_st_handle_slot), allocatable :: tmp(:)
    integer :: i, old_n, new_n

    old_n = size(kriging_st_registry)
    new_n = max(1, old_n * 2)
    allocate(tmp(new_n))
    do i = 1, old_n
      if (associated(kriging_st_registry(i)%obj)) tmp(i)%obj => kriging_st_registry(i)%obj
    end do
    call move_alloc(tmp, kriging_st_registry)
  end subroutine grow_registry

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
