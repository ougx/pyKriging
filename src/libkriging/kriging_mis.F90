!==============================================================================
! Module: kriging_mis
!
! Multiple Indicator Simulation (MIS) extension of t_kriging.
!
! Theory
! ------
! For a continuous variable Z(x) with K thresholds z_1 < z_2 < ... < z_K,
! the indicator transform at threshold k is:
!
!   I(x; z_k) = 1  if Z(x) <= z_k,   0  otherwise
!
! E[I(x; z_k)] = F(z_k) = Prob[Z(x) <= z_k]  (global CDF at threshold k)
!
! Each indicator has its own variogram gamma_k(h).  Sequential indicator
! simulation visits each node in a random order, estimates the local
! conditional CDF via indicator kriging at K thresholds, corrects order
! relations if violated, draws a uniform deviate, and inverts the ccdf to
! obtain a simulated value.  Previously simulated indicator values are added
! to the conditioning set at each step.
!
! For categorical variables (K mutually exclusive classes), the same
! algorithm applies: the K indicator values sum to 1 everywhere, and the
! draw selects a class rather than inverting a ccdf.
!
! Design
! ------
! t_kriging_mis extends t_kriging.  The base class holds ONE vgm_struct per
! variable pair; MIS needs K independent vgm_structs, one per threshold.
!
! New fields vs the base class:
!
!   ivgm(K)          : K indicator variogram models
!   thresh(K)        : K threshold values z_k (continuous) or class indices
!   prop(K)          : global proportions F(z_k) = mean(I(xi; z_k))
!   iobs(K, nobs)    : indicator transforms of all observations
!   mis_sim_iobs(K, nblock) : indicator values for all previously simulated
!                             nodes; grows as simulation proceeds
!   saved_obs_val(nobs)     : copy of original obs(1)%value before solve_mis
!                             modifies it per-threshold
!   is_categorical   : if .true., treat as K-class categorical simulation
!
! Matrix assembly
! ---------------
! The indicator kriging matrix has EXACTLY the same block structure as
! ordinary kriging with nvar=1.  No cross-indicator covariance terms are
! included (the K systems are solved independently).  The base class
! assemble_linear_system and solve_linear_system are reused without any
! modification — only the variogram pointer self%vgm(1,1) is redirected to
! ivgm(k) before each of the K solves, and obs(1)%value is temporarily
! loaded with the k-th indicator column.
!
! This is the same pointer-redirection trick used in t_kriging_sva, applied
! inside the threshold loop rather than the block loop.
!
! Thread safety
! -------------
! MIS requires sequential block processing (each block conditions on
! previously simulated values), so the OMP parallel block loop is disabled
! exactly as in the base-class SGS path (nsim > 0).  The K inner solves per
! block are sequential; parallelising across thresholds within a block is
! possible but not implemented here because K is typically small (9-15).
!
! Usage
! -----
!   type(t_kriging_mis) :: k
!   call k%initialize(ndim=2, nvar=1, nsim=1, unbias=0)   ! SK-based IK
!   call k%set_obs(ivar=1, coord=obs_coord, value=obs_val, nmax=20)
!   call k%set_grid(coord=grid_coord)
!   call k%set_sim()
!   call k%set_thresholds([0.1, 0.3, 0.5, 0.7, 0.9], &
!                          prop=[0.10, 0.25, 0.50, 0.75, 0.90])
!   call k%set_ivgm(k=1, spec="sph 0.0 1.0 300.0 300.0 300.0 0.0 0.0 0.0")
!   call k%set_ivgm(k=2, spec="sph 0.0 1.0 400.0 400.0 400.0 0.0 0.0 0.0")
!   ...
!   call k%compute_iobs()    ! transform obs to indicators; call after set_obs
!   call k%set_search(ivar=1)
!   call k%solve()           ! dispatches to solve_mis
!
!==============================================================================
module kriging_mis

  use variogram,  only: vgm_struct
  use kriging,    only: t_kriging, t_kriging_ctx
  use common,     only: verylarge, zero, one

  implicit none
  private

  public :: t_kriging_mis

  !============================================================================
  type, extends(t_kriging) :: t_kriging_mis

    !-- Indicator variograms, one per threshold.
    integer                          :: nthresh = 0
    type(vgm_struct), allocatable    :: ivgm(:)              ! (nthresh)

    !-- Thresholds and global proportions.
    real,             allocatable    :: thresh(:)             ! (nthresh) z_k values
    real,             allocatable    :: prop(:)               ! (nthresh) F(z_k)

    !-- Indicator transforms of original observations: I(xi; z_k).
    !   Dimensions: (nthresh, nobs1) — filled by compute_iobs().
    real,             allocatable    :: iobs(:,:)             ! (nthresh, nobs)

    !-- Indicator values for previously simulated nodes.
    !   Dimensions: (nthresh, nblock) — column ib filled after simulating ib.
    !   Loaded into obs(1)%value and obs(0)%value before each k-solve.
    real,             allocatable    :: mis_sim_iobs(:,:)     ! (nthresh, nblock)

    !-- Copy of original obs(1)%value before solve_mis modifies it.
    real,             allocatable    :: saved_obs_val(:)      ! (nobs)

    !-- If .true.: K mutually exclusive classes; draw selects class index.
    logical                          :: is_categorical = .false.

    !-- Order-relation correction strategy:
    !   'clamp'     : clamp each ccdf value to [ccdf(k-1), ccdf(k+1)]
    !   'isotonic'  : pool-adjacent-violators (PAV) isotonic regression
    character(8)                     :: order_correction = 'isotonic'

    !-- Tail extrapolation: 0 = linear tails; > 0 = power-law tails.
    real                             :: tail_power = 0.0

  contains
    procedure :: set_thresholds
    procedure :: set_ivgm          => mis_set_ivgm
    procedure :: compute_iobs
    procedure :: solve             => solve_mis
    procedure :: finalize          => finalize_mis
  end type t_kriging_mis

contains

  !============================================================================
  ! set_thresholds
  !
  ! Define the K threshold values and corresponding global proportions.
  ! Must be called before compute_iobs() and solve().
  !============================================================================
  subroutine set_thresholds(self, thresholds, prop)
    class(t_kriging_mis), intent(inout) :: self
    real,                 intent(in)    :: thresholds(:)
    real,                 intent(in)    :: prop(:)

    integer :: k

    if (size(thresholds) /= size(prop)) &
      error stop 't_kriging_mis%set_thresholds: thresholds and prop must have the same length'
    if (size(thresholds) < 2) &
      error stop 't_kriging_mis%set_thresholds: need at least 2 thresholds'

    do k = 2, size(thresholds)
      if (thresholds(k) <= thresholds(k-1)) &
        error stop 't_kriging_mis%set_thresholds: thresholds must be strictly increasing'
      if (prop(k) <= prop(k-1)) &
        error stop 't_kriging_mis%set_thresholds: prop must be strictly increasing'
    end do
    if (prop(1) <= 0.0 .or. prop(size(prop)) >= 1.0) &
      error stop 't_kriging_mis%set_thresholds: require 0 < prop(1) and prop(K) < 1'

    self%nthresh = size(thresholds)
    if (allocated(self%thresh)) deallocate(self%thresh)
    if (allocated(self%prop))   deallocate(self%prop)
    if (allocated(self%ivgm))   deallocate(self%ivgm)
    allocate(self%thresh(self%nthresh), source=thresholds)
    allocate(self%prop  (self%nthresh), source=prop)
    allocate(self%ivgm  (self%nthresh))
  end subroutine set_thresholds


  !============================================================================
  ! mis_set_ivgm  (bound as set_ivgm)
  !
  ! Add one nested structure to the variogram for threshold k.
  ! May be called multiple times for nested models.
  !============================================================================
  subroutine mis_set_ivgm(self, k, spec)
    class(t_kriging_mis), intent(inout) :: self
    integer,              intent(in)    :: k
    character(*),         intent(in)    :: spec

    if (.not. allocated(self%ivgm)) &
      error stop 't_kriging_mis%set_ivgm: call set_thresholds() first'
    if (k < 1 .or. k > self%nthresh) then
      write(*, '(A,I0,A,I0)') &
        't_kriging_mis%set_ivgm: k=', k, ' out of range 1..', self%nthresh
      error stop
    end if
    call self%ivgm(k)%add(spec=spec)
  end subroutine mis_set_ivgm


  !============================================================================
  ! compute_iobs
  !
  ! Compute the indicator transform of all observations.
  ! Must be called AFTER set_obs() and set_thresholds().
  !
  ! Sets iobs(k,i) = 1 if obs(1)%value(i) <= thresh(k), else 0.
  ! Overwrites prop(k) with the sample mean of iobs(k,:) unless
  ! user_prop=.true. is passed.
  !============================================================================
  subroutine compute_iobs(self, user_prop)
    class(t_kriging_mis), intent(inout) :: self
    logical, optional,    intent(in)    :: user_prop

    integer :: k, i, nobs
    logical :: keep_prop

    keep_prop = .false.
    if (present(user_prop)) keep_prop = user_prop

    if (.not. allocated(self%ivgm)) &
      error stop 't_kriging_mis%compute_iobs: call set_thresholds() first'
    if (.not. associated(self%obs)) &
      error stop 't_kriging_mis%compute_iobs: call set_obs() first'

    nobs = self%obs(1)%n
    if (allocated(self%iobs)) deallocate(self%iobs)
    allocate(self%iobs(self%nthresh, nobs))

    do k = 1, self%nthresh
      do i = 1, nobs
        self%iobs(k, i) = merge(one, zero, self%obs(1)%value(i) <= self%thresh(k))
      end do
      if (.not. keep_prop) &
        self%prop(k) = sum(self%iobs(k,:)) / real(nobs)
    end do

    !-- Warn about degenerate thresholds
    do k = 1, self%nthresh
      if (self%prop(k) <= 0.01 .or. self%prop(k) >= 0.99) &
        write(*, '(A,I0,A,F7.4,A,F8.3)') &
          'WARNING t_kriging_mis: threshold k=', k, &
          ' has extreme proportion ', self%prop(k), &
          ' for z_k=', self%thresh(k)
    end do
  end subroutine compute_iobs


  !============================================================================
  ! solve_mis  (overrides t_kriging%solve)
  !
  ! Sequential indicator simulation main loop.
  !============================================================================
  subroutine solve_mis(self)
    use progress_bar, only: progress
    class(t_kriging_mis), intent(inout) :: self

    type(t_kriging_ctx), allocatable :: ctx
    type(vgm_struct),    pointer     :: vgm_orig
    integer  :: ib, k, nobs_orig
    real     :: ccdf(self%nthresh), u, z_sim
    real, allocatable :: temp(:,:)
    character(len=64) :: errmsg

    errmsg = 't_kriging_mis%solve_mis: '

    !-- Pre-flight checks
    if (self%nthresh == 0) &
      error stop trim(errmsg)//'call set_thresholds() before solve()'
    if (.not. allocated(self%iobs)) &
      error stop trim(errmsg)//'call compute_iobs() before solve()'
    do k = 1, self%nthresh
      if (self%ivgm(k)%nstruct == 0) then
        write(*, '(A,I0,A)') trim(errmsg)//'ivgm(', k, ') has no structures; &
          &call set_ivgm(k, spec) for every threshold'
        error stop
      end if
    end do
    if (self%nsim /= 1) &
      error stop trim(errmsg)//'nsim must equal 1; generate ensembles by repeated calls'

    !-- Validate LMC if using a single shared variogram for all thresholds
    !   (not required; each threshold may have its own model)

    nobs_orig = self%obs(1)%n

    !-- Save original obs(1)%value; the inner loop overwrites it per threshold
    if (allocated(self%saved_obs_val)) deallocate(self%saved_obs_val)
    allocate(self%saved_obs_val(nobs_orig), source=self%obs(1)%value)

    !-- Allocate scratch for simulated indicator values (K x nblock)
    if (allocated(self%mis_sim_iobs)) deallocate(self%mis_sim_iobs)
    allocate(self%mis_sim_iobs(self%nthresh, self%block%n))
    self%mis_sim_iobs = 0.0

    !-- Use ivgm(1) as the representative model for prepare()
    vgm_orig => self%vgm(1,1)
    self%vgm(1,1) => self%ivgm(1)
    call self%prepare()
    self%vgm(1,1) => vgm_orig

    !-- Extend obs(0) capacity to hold up to nblock simulated nodes
    !   (base class allocates obs(0) for SGSIM; capacity = nblock per variable)
    !   obs(0)%coord and obs(0)%value are pre-sized by set_grid -> prepare.
    !   We rely on the base class having done this when nsim=1.

    allocate(ctx)
    call ctx%initialize(self)

    if (self%verbose) print '(A,I0,A)', &
      'Starting MIS loop (K=', self%nthresh, ' thresholds)'

    associate(nb => self%block%n, verbose => self%verbose)

      do ib = 1, nb
        ctx%iblock = ib
        if (verbose) call progress(ib, nb)

        !-- Loop over K thresholds: build ccdf(k) at this node
        do k = 1, self%nthresh

          !-- 1a. Load indicator values into obs(1)%value and obs(0)%value.
          !       obs(1)%value: original observations for threshold k
          self%obs(1)%value(1:nobs_orig) = self%iobs(k, 1:nobs_orig)
          !       obs(0)%value: previously simulated nodes, threshold k
          !       (obs(0)%n = ib-1 at this point)
          if (ib > 1) &
            self%obs(0)%value(1:ib-1) = self%mis_sim_iobs(k, 1:ib-1)

          !-- 1b. Redirect variogram pointer to ivgm(k)
          self%vgm(1,1) => self%ivgm(k)

          !-- 1c. Assemble and solve the IK system
          call self%assemble_linear_system(ctx)
          if (ctx%npp > 1) call self%solve_linear_system(ctx)
          call ctx%assign_weight(self)

          !-- 1d. Compute ccdf(k) = weighted sum of indicator values
          ccdf(k) = compute_ik_estimate(self, ctx, k, nobs_orig, ib)

        end do

        !-- Restore variogram pointer and obs(1)%value
        self%vgm(1,1) => vgm_orig
        self%obs(1)%value(1:nobs_orig) = self%saved_obs_val

        !-- 2. Correct order relations
        call correct_order_relations(self, ccdf)

        !-- 3. Draw U(0,1) from pre-generated N(0,1) sample
        u = normal_to_uniform(self%block%sample(1, ib))

        !-- 4. Invert ccdf -> z_sim
        if (self%is_categorical) then
          z_sim = categorical_draw(self, ccdf, u)
        else
          z_sim = invert_ccdf(self, ccdf, u)
        end if
        z_sim = max(self%bounds(1), min(self%bounds(2), z_sim))

        !-- 5. Store simulated value
        self%block%estimate(1, ib) = z_sim

        !-- 6. Update simulated indicator cache and obs(0) for next blocks
        do k = 1, self%nthresh
          self%mis_sim_iobs(k, ib) = merge(one, zero, z_sim <= self%thresh(k))
        end do
        !   obs(0)%coord is set by the base class via prepare/search; ensure
        !   the simulated location is registered so search_neighbors can find it.
        self%obs(0)%coord(:, ib) = self%block%coord(:, ib)
        self%obs(0)%n = ib

      end do ! block loop

    end associate

    if (self%verbose) then
      print *, ''
      print *, 'MIS completed.'
    end if

    !-- Reorder estimates back to original (un-randomised) block order
    allocate(temp, source=self%block%estimate)
    do ib = 1, self%block%n
      self%block%estimate(:, self%block%order(ib)) = temp(:, ib)
    end do
  end subroutine solve_mis


  !============================================================================
  ! finalize_mis  (overrides t_kriging%finalize)
  !============================================================================
  subroutine finalize_mis(self)
    class(t_kriging_mis), intent(inout) :: self
    if (allocated(self%ivgm))          deallocate(self%ivgm)
    if (allocated(self%thresh))        deallocate(self%thresh)
    if (allocated(self%prop))          deallocate(self%prop)
    if (allocated(self%iobs))          deallocate(self%iobs)
    if (allocated(self%mis_sim_iobs))  deallocate(self%mis_sim_iobs)
    if (allocated(self%saved_obs_val)) deallocate(self%saved_obs_val)
    call self%t_kriging%finalize()
  end subroutine finalize_mis


  !============================================================================
  ! Private helpers
  !============================================================================

  !----------------------------------------------------------------------------
  ! compute_ik_estimate
  !
  ! After weights have been assigned to ctx for the k-th threshold, compute
  ! the indicator kriging estimate:
  !
  !   F*(x0; z_k) = sum_i lambda_i * I(xi; z_k)
  !                 + lambda_0 (SK mean = prop(k) if unbias=0)
  !
  ! Primary variable neighbours (ivar=1) carry the original indicator values;
  ! SGSIM conditioning neighbours (ivar=0) carry previously simulated values
  ! from mis_sim_iobs(k, 1:ib-1).
  !----------------------------------------------------------------------------
  function compute_ik_estimate(self, ctx, k, nobs_orig, ib) result(fk)
    type(t_kriging_mis), intent(in) :: self
    type(t_kriging_ctx), intent(in) :: ctx
    integer,             intent(in) :: k, nobs_orig, ib
    real                            :: fk

    integer :: i, idx
    real    :: w_sum

    fk    = 0.0
    w_sum = 0.0

    !-- Primary variable neighbours
    do i = 1, ctx%nnear(1)
      idx  = ctx%inear(i, 1)
      fk   = fk + ctx%weight(i, 1) * self%iobs(k, idx)
      w_sum = w_sum + ctx%weight(i, 1)
    end do

    !-- SGSIM conditioning neighbours (previously simulated, ivar0=0)
    do i = 1, ctx%nnear(0)
      idx = ctx%inear(i, 0)
      if (idx >= 1 .and. idx < ib) &
        fk = fk + ctx%weight(i, 0) * self%mis_sim_iobs(k, idx)
    end do

    !-- SK correction: prop(k) is the a priori mean of I(x; z_k)
    if (self%unbias == 0) then
      fk = fk + (one - w_sum) * self%prop(k)
    end if

    fk = max(0.0, min(1.0, fk))
  end function compute_ik_estimate


  !----------------------------------------------------------------------------
  ! correct_order_relations
  !
  ! Enforce ccdf(1) <= ccdf(2) <= ... <= ccdf(K).
  !
  ! 'isotonic': pool-adjacent-violators (PAV) algorithm.
  !   Scans left to right; when ccdf(k) > ccdf(k+1) is found, merges the
  !   two values to their mean and checks backward until monotonicity is
  !   restored.  Minimises the sum of squared changes (Barlow et al., 1972).
  !
  ! 'clamp': simple forward-backward clamp.
  !   Forward pass: ccdf(k) = max(ccdf(k), ccdf(k-1))
  !   Backward pass: ccdf(k) = min(ccdf(k), ccdf(k+1))
  !----------------------------------------------------------------------------
  subroutine correct_order_relations(self, ccdf)
    type(t_kriging_mis), intent(in)    :: self
    real,                intent(inout) :: ccdf(:)

    integer :: k, j, istart, iend, bsize
    real    :: bmean

    if (self%order_correction == 'clamp') then
      do k = 2, self%nthresh
        if (ccdf(k) < ccdf(k-1)) ccdf(k) = ccdf(k-1)
      end do
      do k = self%nthresh - 1, 1, -1
        if (ccdf(k) > ccdf(k+1)) ccdf(k) = ccdf(k+1)
      end do
      return
    end if

    !-- PAV isotonic regression
    k = 1
    do while (k <= self%nthresh)
      istart = k
      bsize  = 1
      bmean  = ccdf(k)

      !-- Absorb forward violations
      do while (k < self%nthresh .and. ccdf(k+1) < bmean)
        k     = k + 1
        bsize = bsize + 1
        bmean = bmean + (ccdf(k) - bmean) / bsize
      end do
      iend = k

      !-- Absorb backward violations
      do while (istart > 1 .and. ccdf(istart-1) > bmean)
        istart = istart - 1
        bsize  = bsize + 1
        bmean  = bmean + (ccdf(istart) - bmean) / bsize
      end do

      !-- Write block mean
      do j = istart, iend
        ccdf(j) = bmean
      end do
      k = iend + 1
    end do
  end subroutine correct_order_relations


  !----------------------------------------------------------------------------
  ! invert_ccdf
  !
  ! Invert the corrected ccdf to find z_sim given u ~ U(0,1).
  !
  ! Interior (ccdf(k) <= u < ccdf(k+1)): linear interpolation.
  ! Lower tail (u < ccdf(1)):  linear from (z_min, 0) to (thresh(1), prop(1)).
  ! Upper tail (u > ccdf(K)):  linear from (thresh(K), prop(K)) to (z_max, 1).
  ! If tail_power > 0: power-law extrapolation in both tails.
  !
  ! z_min / z_max are estimated by extending the first/last inter-threshold
  ! interval beyond the data range.
  !----------------------------------------------------------------------------
  function invert_ccdf(self, ccdf, u) result(z_sim)
    type(t_kriging_mis), intent(in) :: self
    real,                intent(in) :: ccdf(:)
    real,                intent(in) :: u
    real                            :: z_sim

    integer :: k
    real    :: z_lo, z_hi, f_lo, f_hi, frac
    real    :: z_min, z_max

    z_min = self%thresh(1) - (self%thresh(2) - self%thresh(1))
    z_max = self%thresh(self%nthresh) + &
            (self%thresh(self%nthresh) - self%thresh(self%nthresh-1))

    !-- Lower tail
    if (u <= ccdf(1)) then
      f_lo = 0.0;       z_lo = z_min
      f_hi = ccdf(1);   z_hi = self%thresh(1)
      if (f_hi - f_lo < tiny(1.0)) then
        z_sim = z_lo
      else if (self%tail_power > 0.0) then
        z_sim = z_lo + (z_hi - z_lo) * (u / f_hi)**(1.0 / self%tail_power)
      else
        z_sim = z_lo + (z_hi - z_lo) * (u - f_lo) / (f_hi - f_lo)
      end if
      return
    end if

    !-- Upper tail
    if (u >= ccdf(self%nthresh)) then
      f_lo = ccdf(self%nthresh);   z_lo = self%thresh(self%nthresh)
      f_hi = 1.0;                  z_hi = z_max
      if (f_hi - f_lo < tiny(1.0)) then
        z_sim = z_hi
      else if (self%tail_power > 0.0) then
        z_sim = z_lo + (z_hi - z_lo) * &
                ((u - f_lo) / (1.0 - f_lo))**(1.0 / self%tail_power)
      else
        z_sim = z_lo + (z_hi - z_lo) * (u - f_lo) / (f_hi - f_lo)
      end if
      return
    end if

    !-- Interior
    do k = 1, self%nthresh - 1
      if (u < ccdf(k+1)) then
        f_lo = ccdf(k);       z_lo = self%thresh(k)
        f_hi = ccdf(k+1);     z_hi = self%thresh(k+1)
        if (f_hi - f_lo < tiny(1.0)) then
          z_sim = 0.5 * (z_lo + z_hi)
        else
          frac  = (u - f_lo) / (f_hi - f_lo)
          z_sim = z_lo + frac * (z_hi - z_lo)
        end if
        return
      end if
    end do

    z_sim = self%thresh(self%nthresh)   ! fallback
  end function invert_ccdf


  !----------------------------------------------------------------------------
  ! categorical_draw
  !
  ! For K-class categorical simulation: select the class whose cumulative
  ! proportion first exceeds the uniform deviate u.
  !----------------------------------------------------------------------------
  function categorical_draw(self, ccdf, u) result(z_sim)
    type(t_kriging_mis), intent(in) :: self
    real,                intent(in) :: ccdf(:)
    real,                intent(in) :: u
    real                            :: z_sim

    integer :: k
    z_sim = self%thresh(self%nthresh)
    do k = 1, self%nthresh
      if (u <= ccdf(k)) then
        z_sim = self%thresh(k)
        return
      end if
    end do
  end function categorical_draw


  !----------------------------------------------------------------------------
  ! normal_to_uniform
  !
  ! Transform N(0,1) to U(0,1) via the normal CDF using the rational
  ! approximation of Hart (1968) as used in GSLIB.
  !----------------------------------------------------------------------------
  elemental function normal_to_uniform(z) result(u)
    real, intent(in) :: z
    real             :: u

    real, parameter :: p0=220.2068679123761,  p1=221.2135961699311
    real, parameter :: p2=112.0792914978709,  p3=33.91286607838300
    real, parameter :: p4=6.373962203531650,  p5=0.7003830644436881
    real, parameter :: p6=0.03526249659989109
    real, parameter :: q0=440.4137358247522,  q1=793.8265125199484
    real, parameter :: q2=637.3336333788311,  q3=296.5642487796737
    real, parameter :: q4=86.78073220294608,  q5=16.06417757920695
    real, parameter :: q6=1.755667163182642,  q7=0.08838834764831844
    real :: za, p, q

    za = abs(z)
    if (za > 6.0) then
      u = 0.0
    else
      p = p0 + za*(p1 + za*(p2 + za*(p3 + za*(p4 + za*(p5 + za*p6)))))
      q = q0 + za*(q1 + za*(q2 + za*(q3 + za*(q4 + za*(q5 + za*(q6+za*q7))))))
      u = p / q * exp(-0.5 * za * za)
    end if
    if (z > 0.0) u = 1.0 - u
  end function normal_to_uniform

end module kriging_mis
