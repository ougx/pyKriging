!==============================================================================
! Module: kriging_sva
!
! Spatially Varying Anisotropy (SVA) extension of t_kriging.
!
! Design
! ------
! The base class t_kriging holds a single global variogram array:
!
!   type(vgm_struct), pointer :: vgm(:,:)   ! (ivar0:nvar, ivar0:nvar)
!
! This subclass overrides it with a per-block array:
!
!   type(vgm_struct), pointer :: vgm_sva(:,:,:)  ! (1:nvar, 1:nvar, nblock) ! ivar0=1 since simulation should not be used with SVA.
!
! At solve time the base class pointer self%vgm is redirected to the slice
! for the current block before every call to assemble_linear_system / 
! calc_covariance, then restored afterwards.  Because all of the covariance
! assembly logic lives in the base class and goes through self%vgm(:,:), no
! base-class code needs to change.
!
! Thread safety
! -------------
! Redirecting self%vgm inside the OMP parallel region would create a race
! condition: two threads would stomp each other's pointer.  The workaround
! used here is to disable OMP parallelism (nsim=0 path) by setting
! self%nsim_override and to override solve() with a sequential loop that
! redirects self%vgm block-by-block.  If you compile without OpenMP this
! is irrelevant.
!
! Memory layout
! -------------
! vgm_sva is (ivar0:nvar, ivar0:nvar, nblock), so for a given block ib the
! slice vgm_sva(:,:,ib) maps directly to the base-class vgm(:,:) pointer.
! Fortran pointer association to an interior section requires the section to
! be contiguous in memory.  Because vgm_struct contains allocatables
! (tab, tab_h, shape) each element is independently heap-allocated, so the
! derived-type array itself IS contiguous (only the element metadata lives
! in the array; the dynamic parts live elsewhere).  The pointer re-target is
! therefore safe.
!
! Usage
! -----
!   type(t_kriging_sva) :: k
!   call k%initialize(ndim=2, nvar=1)
!   call k%set_obs(...)
!   call k%set_grid(...)
!   call k%allocate_sva()                          ! call AFTER set_grid
!   call k%set_vgm_block(ib=1, ivar=1, jvar=1, spec="sph 0 1 500 1000 500 0 0 0")
!   call k%set_vgm_block(ib=2, ivar=1, jvar=1, spec="exp 0 1 300  600 300 0 0 0")
!   ...                                            ! or bulk-fill via set_vgm_block_all
!   call k%set_search(...)
!   call k%solve()
!
!==============================================================================
module kriging_sva

  use variogram,  only: vgm_struct
  use kriging,    only: t_kriging, t_kriging_ctx
  use common,     only: verylarge

  implicit none
  character(len=2048)     :: errmsg

  private

  public :: t_kriging_sva

  !============================================================================
  type, extends(t_kriging) :: t_kriging_sva
    !-- Per-block variogram array.  Allocated after set_grid() via allocate_sva().
    !   Dimensions: (ivar0:nvar, ivar0:nvar, nblock)
    type(vgm_struct), pointer :: vgm_sva(:,:,:) => null()
  contains
    procedure :: allocate_sva
    procedure :: set_vgm_block
    procedure :: set_vgm_block_all
    procedure :: solve      => solve_sva      ! override base solve
    procedure :: finalize   => finalize_sva   ! override base finalize
  end type t_kriging_sva

contains

  !============================================================================
  ! allocate_sva
  !
  ! Allocate the per-block variogram array.  Must be called AFTER set_grid()
  ! (which sets self%block%n) and AFTER initialize() (which sets ivar0/nvar).
  !============================================================================
  subroutine allocate_sva(self)
    class(t_kriging_sva), intent(inout) :: self

    if (.not. associated(self%block)) &
      error stop 't_kriging_sva%allocate_sva: call set_grid() first'
    if (self%block%n == 0) &
      error stop 't_kriging_sva%allocate_sva: block%n == 0; call set_grid() first'
    if (associated(self%vgm_sva)) deallocate(self%vgm_sva)

    allocate(self%vgm_sva(self%ivar0:self%nvar, self%ivar0:self%nvar, self%block%n))
  end subroutine allocate_sva


  !============================================================================
  ! set_vgm_block
  !
  ! Add one nested variogram structure to block ib, pair (ivar, jvar).
  ! Call this once per nested structure (just like the base set_vgm but with
  ! an extra block index).
  !
  ! Parameters
  !   ib   : block index  1..nblock
  !   ivar : row variable index
  !   jvar : column variable index
  !   spec : variogram spec string  "vtype nugget sill a_major a_minor1 a_minor2 az dip plunge"
  !============================================================================
  subroutine set_vgm_block(self, ib, ivar, jvar, spec)
    class(t_kriging_sva), intent(inout) :: self
    integer,              intent(in)    :: ib, ivar, jvar
    character(*),         intent(in)    :: spec
    errmsg = "t_kriging_sva%set_vgm_block: "
    if (.not. associated(self%vgm_sva)) &
      error stop 't_kriging_sva%set_vgm_block: call allocate_sva() first'
    if (ib < 1 .or. ib > self%block%n) &
      error stop 't_kriging_sva%set_vgm_block: ib out of range'

    if (jvar==ivar) then
      call self%vgm(jvar, ivar, ib)%add(spec=spec)
    else if (jvar>ivar) then
      call self%vgm(jvar, ivar, ib)%add(spec=spec)
      call self%vgm(ivar, jvar, ib)%add(spec=spec)
    else
      error stop trim(errmsg)//'jvar must be >= ivar to set the upper triangle of the variogram matrix'
    end if
  end subroutine set_vgm_block


  !============================================================================
  ! set_vgm_block_all
  !
  ! Convenience: assign the same variogram spec to ALL blocks for pair
  ! (ivar, jvar).  Useful for testing or as a fallback when only a subset of
  ! blocks have locally fitted variograms.
  !============================================================================
  subroutine set_vgm_block_all(self, ivar, jvar, spec)
    class(t_kriging_sva), intent(inout) :: self
    integer,              intent(in)    :: ivar, jvar
    character(*),         intent(in)    :: spec

    integer :: ib

    if (.not. associated(self%vgm_sva)) &
      error stop 't_kriging_sva%set_vgm_block_all: call allocate_sva() first'

    do ib = 1, self%block%n
      call self%vgm_sva(jvar, ivar, ib)%add(spec=spec)
      if (jvar /= ivar) call self%vgm_sva(ivar, jvar, ib)%add(spec=spec)
    end do
  end subroutine set_vgm_block_all


  !============================================================================
  ! solve_sva  (overrides t_kriging%solve)
  !
  ! Sequential block loop — no OMP parallelism — that redirects self%vgm to
  ! the per-block slice before each assemble/solve/estimate call.
  !
  ! Why sequential?
  !   The redirection self%vgm => self%vgm_sva(:,:,ib) mutates a shared field
  !   of self.  Running this inside an OMP parallel region would require either
  !   a per-thread copy of the t_kriging_sva object (prohibitive memory cost)
  !   or a critical section around every block (eliminates the speedup).
  !   The bottleneck for SVA is typically the per-block variogram fitting
  !   done before this call; the kriging solve itself is fast once weights
  !   are set up.
  !
  ! If you need OMP here the cleanest solution is to copy vgm_sva(:,:,ib)
  ! into a thread-private t_kriging instance per block.  That refactor is
  ! left to the caller.
  !============================================================================
  subroutine solve_sva(self)
    use progress_bar, only: progress
    class(t_kriging_sva), intent(inout) :: self

    type(t_kriging_ctx), allocatable :: ctx
    integer                          :: ib
    real, allocatable                :: temp(:,:)
    character(len=64)                :: errmsg

    errmsg = 't_kriging_sva%solve_sva: '

    if (.not. associated(self%vgm_sva)) &
      error stop trim(errmsg)//'call allocate_sva() and set_vgm_block*() before solve()'

    !-- Validate that every block has at least one structure for every pair.
    call check_all_blocks_set(self)

    !-- Base-class prepare(): computes nppmax, matsize_max, handles file I/O,
    !   and check Variogram has been set.  We call it here through
    !   the base class.  It reads self%vgm(:,:) (the global pointer), so we
    !   first point that at block 1 as a representative model.
    self%vgm => self%vgm_sva(:,:,1)
    call self%prepare()

    associate(nb => self%block%n, verbose => self%verbose)

      if (verbose) print *, 'Starting SVA Kriging loop (sequential)'

      allocate(ctx)
      call ctx%initialize(self)

      do ib = 1, nb
        ctx%iblock = ib
        if (verbose) call progress(ib, nb)

        !-- Redirect the base-class variogram pointer to this block's model.
        self%vgm => self%vgm_sva(:,:,ib)

        if (self%use_old_weight) then
          call self%read_weight(ctx)
        else
          call self%assemble_linear_system(ctx)
          if (ctx%npp > 1) call self%solve_linear_system(ctx)
          call ctx%assign_weight(self)
        end if

        if (self%store_weight) call self%write_weight(ctx)
        call self%estimate_block(ctx)
        if (self%write_mat)    call ctx%write_matrix(self)
      end do

      if (verbose) print *, ''
      if (verbose) print *, 'SVA Kriging completed.'

      !-- If SGSIM: reorder estimates back to original block order.
      if (self%nsim > 0) then
        allocate(temp, source=self%block%estimate)
        do ib = 1, self%block%n
          self%block%estimate(:, self%block%order(ib)) = temp(:, ib)
        end do
      end if

    end associate

    !-- Leave self%vgm pointing at block 1 as a safe residual state.
    self%vgm => self%vgm_sva(:,:,1)
  end subroutine solve_sva


  !============================================================================
  ! finalize_sva  (overrides t_kriging%finalize)
  !
  ! Nullifies self%vgm before calling the base finalizer to avoid a
  ! double-free: self%vgm currently aliases vgm_sva(:,:,1), not a
  ! separately allocated array, so the base-class "deallocate(self%vgm)"
  ! would free memory that vgm_sva still owns.
  !============================================================================
  subroutine finalize_sva(self)
    class(t_kriging_sva), intent(inout) :: self

    !-- Break the alias so the base finalizer skips its deallocate(self%vgm).
    nullify(self%vgm)

    !-- Release the per-block array.
    if (associated(self%vgm_sva)) deallocate(self%vgm_sva)
    nullify(self%vgm_sva)

    !-- Base finalizer: deallocates obs, grid, block.
    call self%t_kriging%finalize()
  end subroutine finalize_sva


  !============================================================================
  ! check_all_blocks_set  (private helper)
  !
  ! Verify that every block/variable-pair has at least one structure defined.
  ! Stops with a descriptive error if any block is missing its variogram.
  !============================================================================
  subroutine check_all_blocks_set(self)
    type(t_kriging_sva), intent(in) :: self

    integer            :: ib, iv, jv
    character(len=256) :: msg

    do ib = 1, self%block%n
      do iv = self%ivar0, self%nvar
        do jv = self%ivar0, self%nvar
          if (self%vgm_sva(jv, iv, ib)%nstruct == 0) then
            write(msg, '(A,I0,A,I0,A,I0,A)') &
              't_kriging_sva: variogram not set for block ', ib, &
              ', ivar=', iv, ', jvar=', jv, &
              '. Call set_vgm_block() or set_vgm_block_all().'
            error stop trim(msg)
          end if
        end do
      end do
    end do
  end subroutine check_all_blocks_set

end module kriging_sva
