! This is the block Cholesky / Schur complement solver for universal kriging (UK)

! Schur complement
!   | A   B | | x | = | u |
!   | C   D | | y |   | v |
!  x =  A⁻¹ (u - B y)

! Solving sytem below:
!   | K   F | | w | = | k |
!   | Fᵀ  0 | | λ |   | f |
! Define:
!    A = K      (n×n)
!    B = F      (n×p)
! From the first row:
!    A w + B λ = k
! Solve for w:
!    w = A⁻¹ (k − Bλ)
! Substitute into second row:
!    Bᵀ w = f
!    Bᵀ A⁻¹ (k − Bλ) = f
! Expand:
!    Bᵀ A⁻¹ k − Bᵀ A⁻¹ B λ = f
! Rearrange:
!    (Bᵀ A⁻¹ B) λ = Bᵀ A⁻¹ k − f
! Define:
!    S = Bᵀ A⁻¹ B   (p×p)
!    rhs = Bᵀ A⁻¹ k − f
! So:
!    S λ = rhs
! Once λ is known:
!    w = A⁻¹ (k − Bλ)
! We never form A⁻¹. Instead:
!    A = L Lᵀ
!    A Y = B   →   Y = A⁻¹ B
!    A y = k   →   y = A⁻¹ k
! Then:
!    S = Bᵀ Y
!    rhs = Bᵀ y − f
! Solve small system:
!    S λ = rhs
! Then:
!    w = y - Y λ
! Summary (compact form)
!    (1) A = L Lᵀ
!    (2) Y = A⁻¹ B
!    (3) y = A⁻¹ k
!    (4) S = Bᵀ Y
!    (5) λ = S⁻¹ (Bᵀ y − f)
!    (6) w = y − Y λ

! Solving sytem below:
!   | K   F | | w | = | k |
!   | Fᵀ  0 | | λ |   | f |
! Where:
!   K (n x n): covariance matrix (SPD)
!   F (n x p): drift/design matrix (p drift terms)
!     e.g. [1, x, y, z, ...]
!   k (n): covariance vector to estimation point
!   f (p): drift evaluated at estimation point
!   λ (p): Lagrange multipliers
! Step 1. Factor (Cholesky decomposition):
!   K = L Lᵀ
! Step 2. forward Solve:
!   A = C⁻¹ F     (n x p)
!   y = C⁻¹ k     (n)
! Step 3: Build small system:
!   S = Fᵀ A      (p x p)
!   rhs = Fᵀ y - f
! Step 4: Solve:
!   S λ = rhs
! Step 5: Solve:
!   w = y - A λ
!   variance = sill - dot_product(k, y) - mu

! Special case: ordinary kriging
!   B = 1 (vector of ones)
!   p = 1
! Then:
!   S = 1ᵀ A⁻¹ 1   (scalar)
! and:
!   λ = (1ᵀ y − 1) / (1ᵀ z)
!   z = A⁻¹ 1

! Implmentation:
!   call dpotrf   ! factor
!   call dpotrs   ! solve
module solver
character(1), parameter :: uplo='U'
contains
  ! ----- Setup phase (once per neighborhood) -----
  subroutine uk_setup(K, F, L, A, S, n, p, info)
    implicit none

    integer, intent(in)  :: n, p
    real   , intent(in)  :: K(n,n)
    real   , intent(in)  :: F(n,p)

    real   , intent(out) :: L(n,n)
    real   , intent(out) :: A(n,p)
    real   , intent(out) :: S(p,p)

    integer, intent(out) :: info

    !---------------------------------------
    ! Cholesky: K = L L^T
    L = K
    call spotrf(uplo, n, L, n, info)
    if (info /= 0) return ! stop "Cholesky failed"

    !---------------------------------------
    ! A = K^{-1} F  (solve for all columns at once)
    A = F
    call spotrs(uplo, n, p, L, n, A, n, info)
    if (info /= 0) return ! stop "Cholesky solve failed"

    !---------------------------------------
    ! S = F^T A
    S = matmul(transpose(F), A)
  end subroutine


  ! ----- Solve phase (per estimation point) -----
  subroutine uk_solve(L, A, S, k, f, w, lambda, n, p, info)
    implicit none

    integer, intent(in) :: n, p
    real, intent(in) :: L(n,n)
    real, intent(in) :: A(n,p), S(p,p)
    real, intent(in) :: k(n)
    real, intent(in) :: f(p)

    real, intent(out) :: w(n)
    real, intent(out) :: lambda(p)
    integer, intent(out) :: info

    real :: y(n)
    real :: rhs(p)

    !---------------------------------------
    ! y = K^{-1} k
    y = k
    call spotrs(uplo, n, 1, L, n, y, n, info)
    if (info /= 0) return ! stop "Cholesky solve failed"

    !---------------------------------------
    ! rhs = F^T y - f
    rhs = matmul(transpose(A), k)   ! equivalent to F^T y; works because: A = K⁻¹F  ⇒  Fᵀy = (K⁻¹F)ᵀ k = Aᵀ k
    rhs = rhs - f

    !---------------------------------------
    ! Solve S λ = rhs  (small system)
    call sposv(uplo, p, 1, S, p, rhs, p, info)
    if (info /= 0) return ! stop "Small system solve failed"

    lambda = rhs

    !---------------------------------------
    ! w = y - A λ
    w = y - matmul(A, lambda)
  end subroutine

  !
  !-----------------------------------------------------------------------
  ! Universal Kriging via Block-Cholesky / Schur Complement
  !
  ! Solves q right-hand sides simultaneously:
  !     [ K   F ] [ W ] = [ B ]
  !     [ Fᵀ  0 ] [ Λ ]   [ G ]
  !
  ! where:
  !   K (n×n) : covariance matrix (SPD)
  !   F (n×p) : drift/design matrix (p drift terms; p=1 for ordinary kriging)
  !   B (n×q) : covariance vectors to q estimation points
  !   G (p×q) : drift evaluated at q estimation points
  !   q       : number of RHS columns (usually 1)
  !
  ! Outputs:
  !   W (n×q) : kriging weights for each of the q RHS columns
  !   Λ (p×q) : Lagrange multipliers for each of the q RHS columns
  !
  ! Method (same Schur complement for all q simultaneously):
  !   1. Factor K = L Lᵀ  (Cholesky, once)
  !   2. Solve Y = K⁻¹ F               (n×p, once)
  !   3. Solve Z = K⁻¹ B               (n×q)
  !   4. Form Schur complement S = Fᵀ Y (p×p, once)
  !   5. Solve S Λ = Fᵀ Z − G          (p×q)
  !   6. Compute W = Z − Y Λ            (n×q)
  !
  !-----------------------------------------------------------------------
  !
  !-----------------------------------------------------------------------
  ! Universal Kriging via Block-Cholesky / Schur Complement
  !
  ! Solves q right-hand sides using the full augmented system arrays
  ! directly — no temporary sub-array copies at the call site.
  !
  ! System layout in the passed arrays:
  !
  !   matA  (n+p, n+p):    [ K   F  ]    K = covariance (SPD, n×n)
  !                        [ Fᵀ  0  ]    F = drift/design matrix (n×p)
  !
  !   rhsB  (q, n+p):      [ B  | G ]    B = covariance RHS (q×n)
  !                                       G = drift RHS      (q×p)
  !
  !   x     (q, n+p):  solution, same layout as rhsB
  !                        [ W  | Λ ]    W = kriging weights (q×n)
  !                                       Λ = Lagrange multipliers (q×p)
  !
  ! Arguments
  !   n     : data points in the neighbourhood (K block size)
  !   p     : drift/constraint terms (0=simple, 1=ordinary, >1=universal)
  !   q     : number of RHS columns (usually 1)
  !   matA  : (n+p) × (n+p) augmented matrix (column-major, Fortran order)
  !   rhsB  : q × (n+p) augmented RHS  (first index = realisation/column)
  !   x     : q × (n+p) solution        (same index convention as rhsB)
  !   info  : 0 on success; non-zero indicates which step failed
  !
  ! Note on memory layout
  !   Fortran stores arrays column-major.  matA(1:n, 1:n) is a contiguous
  !   leading sub-matrix and can be passed to LAPACK directly.
  !   rhsB(i, 1:n) is a strided row, so we copy it into a local (n,q)
  !   column-major array for the LAPACK solve — one unavoidable copy.
  !-----------------------------------------------------------------------
  subroutine kriging_solve(n, p, q, matA, rhsB, x, info)

    implicit none

    integer, intent(in)  :: n, p, q
    real,    intent(in)  :: matA(n+p, n+p)
    real,    intent(in)  :: rhsB(q, n+p)
    real,    intent(out) :: x(q, n+p)
    integer, intent(out) :: info

    ! -------- locals ----------
    real    :: L(n,n)
    real    :: kinv_drift(n,p)    ! K⁻¹ F  (n×p, column-major for LAPACK)
    real    :: kinv_rhs(n,q)      ! K⁻¹ B  (n×q, column-major for LAPACK)
    real    :: schur_mat(p,p)     ! Fᵀ K⁻¹ F  (p×p)
    real    :: rhs_small(p,q)     ! Schur RHS  (p×q)
    integer :: i

    !--------------------------------------------------
    ! Cholesky factorization  K = L Lᵀ
    ! matA(1:n, 1:n) is contiguous — no copy needed.
    L = matA(1:n, 1:n)
    call spotrf(uplo, n, L, n, info)
    if (info /= 0) return

    !--------------------------------------------------
    ! SIMPLE KRIGING  (p = 0)
    if (p == 0) then
      ! Transpose rhsB rows into column-major kinv_rhs for LAPACK
      do i = 1, q
        kinv_rhs(:, i) = rhsB(i, 1:n)
      end do
      call spotrs(uplo, n, q, L, n, kinv_rhs, n, info)
      if (info /= 0) return
      do i = 1, q
        x(i, 1:n) = kinv_rhs(:, i)
      end do
      return
    end if

    !--------------------------------------------------
    ! UNIVERSAL / ORDINARY KRIGING

    ! kinv_drift = K⁻¹ F  (n×p, shared across all q RHS)
    ! matA(1:n, n+1:n+p) is a contiguous column slice — no copy needed.
    kinv_drift = matA(1:n, n+1:n+p)
    call spotrs(uplo, n, p, L, n, kinv_drift, n, info)
    if (info /= 0) return

    ! kinv_rhs = K⁻¹ B  (n×q)
    ! rhsB rows are strided — must transpose into column-major kinv_rhs.
    do i = 1, q
      kinv_rhs(:, i) = rhsB(i, 1:n)
    end do
    call spotrs(uplo, n, q, L, n, kinv_rhs, n, info)
    if (info /= 0) return

    ! Schur complement  S = Fᵀ K⁻¹ F  (p×p, shared)
    schur_mat = matmul(transpose(matA(1:n, n+1:n+p)), kinv_drift)

    ! Schur RHS:  Fᵀ K⁻¹ B − G  (p×q)
    ! G = rhsB(1:q, n+1:n+p) — strided rows, extract column-by-column.
    rhs_small = matmul(transpose(matA(1:n, n+1:n+p)), kinv_rhs)
    do i = 1, q
      rhs_small(:, i) = rhs_small(:, i) - rhsB(i, n+1:n+p)
    end do

    ! Solve  S Λ = rhs_small  (p×q small system)
    call sposv(uplo, p, q, schur_mat, p, rhs_small, p, info)
    if (info /= 0) return

    ! Write weights and multipliers back into x
    do i = 1, q
      x(i, 1:n)     = kinv_rhs(:, i) - matmul(kinv_drift, rhs_small(:, i))
      x(i, n+1:n+p) = rhs_small(:, i)
    end do
  end subroutine


  subroutine gaussian_elimination(n, p, q, matA, rhsB, x, info)
    ! Fallback full-system solver with partial pivoting.
    ! Same array layout as kriging_solve: matA(n+p,n+p), rhsB(q,n+p), x(q,n+p).
    implicit none

    integer, intent(in)  :: n, p, q
    real,    intent(in)  :: matA(n+p, n+p)
    real,    intent(in)  :: rhsB(q, n+p)
    real,    intent(out) :: x(q, n+p)
    integer, intent(out) :: info

    integer :: m
    real, allocatable :: A(:,:), R(:,:)
    integer :: k, piv, i, j
    real    :: maxv, factor

    m    = n + p
    info = 0
    allocate(A(m,m), R(m,q))

    ! Copy into column-major locals for in-place elimination
    A = matA
    do i = 1, q
      R(:, i) = rhsB(i, :)
    end do

    ! Forward elimination with partial pivoting
    do k = 1, m-1
      piv  = k
      maxv = abs(A(k,k))
      do i = k+1, m
        if (abs(A(i,k)) > maxv) then
          maxv = abs(A(i,k)); piv = i
        end if
      end do
      if (piv /= k) then
        A([k,piv], :) = A([piv,k], :)
        R([k,piv], :) = R([piv,k], :)
      end if
      if (abs(A(k,k)) < 1.0e-10) then
        info = k; return
      end if
      do i = k+1, m
        factor    = A(i,k) / A(k,k)
        A(i,k:m)  = A(i,k:m) - factor * A(k,k:m)
        R(i,:)    = R(i,:)   - factor * R(k,:)
      end do
    end do

    ! Back substitution for all q columns
    do j = 1, q
      do i = m, 1, -1
        if (i < m) R(i,j) = R(i,j) - sum(A(i,i+1:m) * R(i+1:m,j))
        R(i,j) = R(i,j) / A(i,i)
      end do
      x(j, :) = R(:, j)
    end do
  end subroutine
end module
