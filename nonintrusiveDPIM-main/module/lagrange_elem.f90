! **********************************************************************
! ********************** LAGRANGE ELEMENT MODULE ***********************
! **********************************************************************
!   collection of subroutines to perform element realted calculations
!            subroutines are arranged in alphabetical order.
! **********************************************************************

module lagarange_elem

! **********************************************************************
! **********************************************************************
! interfaces to evaluate shape functions from the interpolation submodule
! **********************************************************************
! **********************************************************************
  interface

  module subroutine interpFunc1(nNode,nInt,intPt,xi_inpt,Nxi,dNdxi)
    use global_parameters
    implicit none
    integer, intent(in)         :: nNode, nInt, intpt
    real(kind=wp), intent(in)   :: xi_inpt(nInt,1)
    real(kind=wp), intent(out)  :: Nxi(nNode), dNdxi(nNode,1)
  end subroutine


  module subroutine interpFunc2(nNode,nInt,intPt,xi_inpt,Nxi,dNdxi)
    use global_parameters
    implicit none    
    integer, intent(in)         :: nNode, nInt, intpt
    real(kind=wp), intent(in)   :: xi_inpt(nInt,1)
    real(kind=wp), intent(out)  :: Nxi(nNode), dNdxi(nNode,1)
  end subroutine


  module subroutine interpFunc3(nNode,nInt,intPt,xi_inpt,Nxi,dNdxi)
    use global_parameters
    implicit none
    integer, intent(in)         :: nNode, nInt, intpt
    real(kind=wp), intent(in)   :: xi_inpt(nInt,1)
    real(kind=wp), intent(out)  :: Nxi(nNode), dNdxi(nNode,1)
  end subroutine


  module subroutine faceNodes(nDim,nNode,face,list,nFaceNodes)
    implicit none
    integer, intent(in)   :: nDim, nNode, face
    integer, intent(out)  :: list(*)
    integer, intent(out)  :: nFaceNodes
  end subroutine

  end interface

! **********************************************************************
! **********************************************************************
! interfaces to perform Gauss integration from the quadrature submodule
! **********************************************************************
! **********************************************************************

  interface

  module subroutine getGaussQuadrtr1(nNode,nInt,w,xi)
    use global_parameters
    implicit none
    integer, intent(in)         :: nNode, nInt
    real(kind=wp), intent(out)  :: w(nInt), xi(nInt,1)
  end subroutine


  module subroutine getGaussQuadrtr2(nNode,nInt,w,xi)
    use global_parameters
    implicit none
    integer, intent(in)         :: nNode, nInt
    real(kind=wp), intent(out)  :: w(nInt), xi(nInt,1)
  end subroutine


  module subroutine getGaussQuadrtr3(nNode,nInt,w,xi)
    use global_parameters
    implicit none
    integer, intent(in)         :: nNode, nInt
    real(kind=wp), intent(out)  :: w(nInt), xi(nInt,1)
  end subroutine

  end interface

end module lagrange_elem


! **********************************************************************
! ***************** INTERPOLATION FUNCTION SUBMODULE *******************
! **********************************************************************
!  available elements:    (a) 1D bar/truss element (2 and 3 nodes)
!                         (b) 2D tri elements (3 and 6 nodes)
!                         (c) 2D quad elements (4 and 8 nodes)
!                         (d) 3D tet elements (4 and 10 nodes)
!                         (e) 3D hex elements (8 and 20 nodes)
! **********************************************************************
! **********************************************************************

submodule (lagarange_elem) interpolation

  contains

! **********************************************************************
module subroutine interpFunc1(nNode,nInt,intPt,xi_int,Nxi,dNdxi)

    use global_parameters
    implicit none

    integer:: nNode, nInt, intpt

    double precision:: xi, xi_int(nInt,1), Nxi(nNode), dNdxi(nNode,1)

    xi    = xi_int(intpt,1)

    if (nNode .eq. 2) then      ! 2 node linear bar element
      ! shape functions
      Nxi(1) = half*(one - xi)
      Nxi(2) = half*(one + xi)

      ! the first derivatives of the shape functions dN/dxi (2x1)
      dNdxi(1,1)  = -half
      dNdxi(2,1)  = half

    elseif (nNode .eq. 3) then  ! 3 node quadratic bar element
      ! shape functions
      Nxi(1)  = -half*xi*(one - xi)
      Nxi(2)  = one-xi**two
      Nxi(3)  = half*xi*(one + xi)

      ! the first derivatives of the shape functions dN/dxi (3x1)
      dNdxi(1,1)  = -half+xi
      dNdxi(2,1)  = -two*xi
      dNdxi(3,1)  = half+xi

    else
      write(15,*) 'element is not supported for 1D analysis', nNode
      write(*,*)  'element is not supported for 1D analysis', nNode
      return
    endif

    return

  end subroutine interpFunc1

! **********************************************************************

module subroutine interpFunc2(nNode,nInt,intPt,xi_int,Nxi,dNdxi)

  ! this subroutine calculates shape function of 2D elements at gauss pts
  ! available 2D elements are: 3 node tri, 6 node tri, 4 node quad, 8 node quad

  ! Nxi(i)          = shape function of node i at the intpt.
  ! dNdxi(i,j)      = derivative wrt j direction of shape fn of node i

    use global_parameters
    implicit none

    integer:: nNode, nInt, intpt

    double precision:: xi_int(nInt,2), Nxi(nNode), dNdxi(nNode,2)
    double precision:: xi, eta, lam

    ! location in the master element
    xi    = xi_int(intpt,1)
    eta   = xi_int(intpt,2)

    !              A eta (=xi_2)
    !              |
    !              3
    !              |\
    !              | \
    !              |  \
    !              |   \
    !              |    \
    !              |     \
    !              |      \
    !              |       \
    !              1--------2--> xi (=xi_1)

    if (nNode.eq.3) then        ! 3-noded tri3 linear element
      ! shape functions
      Nxi(1) = xi
      Nxi(2) = eta
      Nxi(3) = one - xi - eta

      ! the first derivatives of the shape functions dN/dxi (3x2)
      dNdxi(1, 1) = one
      dNdxi(1, 2) = zero
      dNdxi(2, 1) = zero
      dNdxi(2, 2) = one
      dNdxi(3, 1) = -one
      dNdxi(3, 2) = -one

    !              A eta (=xi_2)
    !              |
    !              3
    !              |\
    !              | \
    !              |  \
    !              |   5
    !              6    \
    !              |     \
    !              |      \
    !              |       \
    !              1---4----2--> xi (=xi_1)

    elseif (nNode.eq.6) then    ! 6-noded quadratic tri6 element
      ! shape functions
      lam = one - xi - eta
      Nxi(1) = lam*(two*lam - one)
      Nxi(2) = xi*(two*xi - one)
      Nxi(3) = eta*(two*eta - one)
      Nxi(4) = four*xi*lam
      Nxi(5) = four*xi*eta
      Nxi(6) = four*eta*lam

      ! the first derivatives of the shape functions dN/dxi (6x2)
      dNdxi(1,1) = one - four*lam
      dNdxi(1,2) = one - four*lam
      dNdxi(2,1) = four*xi - one
      dNdxi(2,2) = zero
      dNdxi(3,1) = zero
      dNdxi(3,2) = four*eta - one
      dNdxi(4,1) = four*(lam - xi)
      dNdxi(4,2) = -four*xi
      dNdxi(5,1) = four*eta
      dNdxi(5,2) = four*xi
      dNdxi(6,1) = -four*eta
      dNdxi(6,2) = four*(lam - eta)

    !                          eta
    !   4-----------3          |
    !   |           |          |
    !   |           |          |
    !   |           |          O--------- xi
    !   |           |
    !   |           |        origin at center
    !   1-----------2

    elseif (nNode.eq.4) then    ! 4-noded bilinear quad4 element
      ! shape functions
      Nxi(1) = fourth*(one - xi)*(one - eta)
      Nxi(2) = fourth*(one + xi)*(one - eta)
      Nxi(3) = fourth*(one + xi)*(one + eta)
      Nxi(4) = fourth*(one - xi)*(one + eta)

      ! the first derivatives of the shape functions dN/dxi (4x2)
      dNdxi(1,1) = -fourth*(one - eta)
      dNdxi(1,2) = -fourth*(one - xi)
      dNdxi(2,1) = fourth*(one - eta)
      dNdxi(2,2) = -fourth*(one + xi)
      dNdxi(3,1) = fourth*(one + eta)
      dNdxi(3,2) = fourth*(one + xi)
      dNdxi(4,1) = -fourth*(one + eta)
      dNdxi(4,2) = fourth*(one - xi)

    !         A eta
    !         |
    !         |
    !   4-----7-----3
    !   |     |     |
    !   |     |     |
    !   8     ------6---> xi
    !   |           |
    !   |           |
    !   1-----5-----2

    elseif (nNode.eq.8) then    ! 8-noded serendipity quad8 element
      ! shape functions
      Nxi(1) = -fourth*(one - xi)*(one - eta)*(one + xi + eta)
      Nxi(2) = -fourth*(one + xi)*(one - eta)*(one - xi + eta)
      Nxi(3) = -fourth*(one + xi)*(one + eta)*(one - xi - eta)
      Nxi(4) = -fourth*(one - xi)*(one + eta)*(one + xi - eta)
      Nxi(5) = half*(one - xi)*(one + xi)*(one - eta)
      Nxi(6) = half*(one - eta)*(one + eta)*(one + xi)
      Nxi(7) = half*(one - xi)*(one + xi)*(one + eta)
      Nxi(8) = half*(one - eta)*(one + eta)*(one - xi)

      ! the first derivatives of the shape functions dN/dxi (8x2)
      dNdxi(1,1) = fourth*(one - eta)*(two*xi + eta)
      dNdxi(1,2) = fourth*(one - xi)*(two*eta + xi)
      dNdxi(2,1) = fourth*(one - eta)*(two*xi - eta)
      dNdxi(2,2) = fourth*(one + xi)*(two*eta - xi)
      dNdxi(3,1) = fourth*(one + eta)*(two*xi + eta)
      dNdxi(3,2) = fourth*(one + xi)*(two*eta + xi)
      dNdxi(4,1) = fourth*(one + eta)*(two*xi - eta)
      dNdxi(4,2) = fourth*(one - xi)*(two*eta - xi)
      dNdxi(5,1) = -xi*(one - eta)
      dNdxi(5,2) = -half*(one - xi)*(one + xi)
      dNdxi(6,1) = half*(one - eta)*(one + eta)
      dNdxi(6,2) = -eta*(one + xi)
      dNdxi(7,1) = -xi*(one + eta)
      dNdxi(7,2) = half*(one - xi)*(one + xi)
      dNdxi(8,1) = -half*(one - eta)*(one + eta)
      dNdxi(8,2) = -eta*(one - xi)

    else
      write(15,*) 'element is not supported for 2D analysis', nNode
      write(*,*)  'element is not supported for 2D analysis', nNode
      return
    endif

    return

  end subroutine interpFunc2

! **********************************************************************

module subroutine interpFunc3(nNode,nInt,intPt,xi_int,Nxi,dNdxi)

    ! this subroutine calculates shape function of 3D elements at gauss pts
    ! available 3D elements are: 4 node tet, 10 node ter, 8 node hex.

    ! Nxi(i)          = shape function of node i at the intpt.
    ! dNdxi(i,j)      = derivative wrt j direction of shape fn of node i

    use global_parameters
    implicit none

    integer:: nNode, nInt, intpt

    double precision:: xi_int(nInt,3), Nxi(nNode), dNdxi(nNode,3)
    double precision:: xi, eta, zeta, lam

    ! Nxi(i)          = shape function of node i at the intpt.
    ! dNdxi(i,j)      = derivative wrt j direction of shape fn of node i

    ! location in the master element
    xi = xi_int(intpt,1)
    eta = xi_int(intpt,2)
    zeta = xi_int(intpt,3)

    Nxi = zero
    dNdxi = zero

    if (nNode.eq.4) then      ! 4-noded linear tet4 element
      ! shape functions
      Nxi(1) = one-xi-eta-zeta
      Nxi(2) = xi
      Nxi(3) = eta
      Nxi(4) = zeta
      ! the first derivatives of the shape functions dN/dxi (4x3)
      dNdxi(1,1) = -one
      dNdxi(1,2) = -one
      dNdxi(1,3) = -one
      dNdxi(2,1) = one
      dNdxi(3,2) = one
      dNdxi(4,3) = one

    elseif (nNode.eq.10) then  ! 10-noded quadratic tet10 element

      ! shape functions
      lam = one-xi-eta-zeta

      Nxi(1) = (two*lam-one)*lam
      Nxi(2) = (two*xi-one)*xi
      Nxi(3) = (two*eta-one)*eta
      Nxi(4) = (two*zeta-one)*zeta
      Nxi(5) = four*xi*lam
      Nxi(6) = four*xi*eta
      Nxi(7) = four*eta*lam
      Nxi(8) = four*zeta*lam
      Nxi(9) = four*zeta*xi
      Nxi(10) = four*eta*zeta

      ! first derivative of shape functions dN/dxi (10x3)
      dNdxi(1,1) = -(four*lam-one)
      dNdxi(1,2) = -(four*lam-one)
      dNdxi(1,3) = -(four*lam-one)
      dNdxi(2,1) = (four*xi-one)
      dNdxi(3,2) = (four*eta-one)
      dNdxi(4,3) = (four*zeta-one)
      dNdxi(5,1) = four*(lam-xi)
      dNdxi(5,2) = -four*xi
      dNdxi(5,3) = -four*xi
      dNdxi(6,1) = four*eta
      dNdxi(6,2) = four*xi
      dNdxi(7,1) = -four*eta
      dNdxi(7,2) = four*(lam-eta)
      dNdxi(7,3) = -four*eta
      dNdxi(8,1) = -four*zeta
      dNdxi(8,2) = -four*zeta
      dNdxi(8,3) = four*(lam-zeta)
      dNdxi(9,1) = four*zeta
      dNdxi(9,3) = four*xi
      dNdxi(10,2) = four*zeta
      dNdxi(10,3) = four*eta

    !      8-----------7
    !     /|          /|
    !    / |         / |      zeta
    !   5-----------6  |       |   eta
    !   |  |        |  |       |   /
    !   |  |        |  |       |  /
    !   |  4--------|--3       | /
    !   | /         | /        |/
    !   |/          |/         O--------- xi
    !   1-----------2        origin at cube center

    elseif(nNode.eq.8) then   ! 8-noded trilinear hex8 element

      ! shape functions
      Nxi(1) = eighth*(one - xi)*(one - eta)*(one - zeta)
      Nxi(2) = eighth*(one + xi)*(one - eta)*(one - zeta)
      Nxi(3) = eighth*(one + xi)*(one + eta)*(one - zeta)
      Nxi(4) = eighth*(one - xi)*(one + eta)*(one - zeta)
      Nxi(5) = eighth*(one - xi)*(one - eta)*(one + zeta)
      Nxi(6) = eighth*(one + xi)*(one - eta)*(one + zeta)
      Nxi(7) = eighth*(one + xi)*(one + eta)*(one + zeta)
      Nxi(8) = eighth*(one - xi)*(one + eta)*(one + zeta)

      ! the first derivatives: dN/dxi (8x3)
      dNdxi(1,1) = -eighth*(one - eta)*(one - zeta)
      dNdxi(1,2) = -eighth*(one - xi)*(one - zeta)
      dNdxi(1,3) = -eighth*(one - xi)*(one - eta)
      dNdxi(2,1) = eighth*(one - eta)*(one - zeta)
      dNdxi(2,2) = -eighth*(one + xi)*(one - zeta)
      dNdxi(2,3) = -eighth*(one + xi)*(one - eta)
      dNdxi(3,1) = eighth*(one + eta)*(one - zeta)
      dNdxi(3,2) = eighth*(one + xi)*(one - zeta)
      dNdxi(3,3) = -eighth*(one + xi)*(one + eta)
      dNdxi(4,1) = -eighth*(one + eta)*(one - zeta)
      dNdxi(4,2) = eighth*(one - xi)*(one - zeta)
      dNdxi(4,3) = -eighth*(one - xi)*(one + eta)
      dNdxi(5,1) = -eighth*(one - eta)*(one + zeta)
      dNdxi(5,2) = -eighth*(one - xi)*(one + zeta)
      dNdxi(5,3) = eighth*(one - xi)*(one - eta)
      dNdxi(6,1) = eighth*(one - eta)*(one + zeta)
      dNdxi(6,2) = -eighth*(one + xi)*(one + zeta)
      dNdxi(6,3) = eighth*(one + xi)*(one - eta)
      dNdxi(7,1) = eighth*(one + eta)*(one + zeta)
      dNdxi(7,2) = eighth*(one + xi)*(one + zeta)
      dNdxi(7,3) = eighth*(one + xi)*(one + eta)
      dNdxi(8,1) = -eighth*(one + eta)*(one + zeta)
      dNdxi(8,2) = eighth*(one - xi)*(one + zeta)
      dNdxi(8,3) = eighth*(one - xi)*(one + eta)

    !
    !       8-----15----- 7
    !      /|            /|
    !    16 |          14 |
    !    /  20        /   |     zeta
    !   5-----13-----6   19      |     eta
    !   |   |        |    |      |    /
    !   |   |        |    |      |   /
    !   17  4-----11-|----3      |  /
    !   |  /         18  /       | /
    !   | 12         | 10        |/
    !   |/           |/          O--------- xi
    !   1-----9------2        origin at cube center
    !
    ! mid-side nodes are not properly illustrated

    elseif (nNode.eq.20) then   ! 20-noded serendipity hex20 element
      ! shape functions
      Nxi(1) = (one-xi)*(one-eta)*(one-zeta)*(-xi-eta-zeta-two)/eight
      Nxi(2) = (one+xi)*(one-eta)*(one-zeta)*(xi-eta-zeta-two)/eight
      Nxi(3) = (one+xi)*(one+eta)*(one-zeta)*(xi+eta-zeta-two)/eight
      Nxi(4) = (one-xi)*(one+eta)*(one-zeta)*(-xi+eta-zeta-two)/eight
      Nxi(5) = (one-xi)*(one-eta)*(one+zeta)*(-xi-eta+zeta-two)/eight
      Nxi(6) = (one+xi)*(one-eta)*(one+zeta)*(xi-eta+zeta-two)/eight
      Nxi(7) = (one+xi)*(one+eta)*(one+zeta)*(xi+eta+zeta-two)/eight
      Nxi(8) = (one-xi)*(one+eta)*(one+zeta)*(-xi+eta+zeta-two)/eight
      Nxi(9)  = (one-xi**two)*(one-eta)*(one-zeta)/four
      Nxi(10) = (one+xi)*(one-eta**two)*(one-zeta)/four
      Nxi(11) = (one-xi**two)*(one+eta)*(one-zeta)/four
      Nxi(12) = (one-xi)*(one-eta**two)*(one-zeta)/four
      Nxi(13) = (one-xi**two)*(one-eta)*(one+zeta)/four
      Nxi(14) = (one+xi)*(one-eta**two)*(one+zeta)/four
      Nxi(15) = (one-xi**two)*(one+eta)*(one+zeta)/four
      Nxi(16) = (one-xi)*(one-eta**two)*(one+zeta)/four
      Nxi(17) = (one-xi)*(one-eta)*(one-zeta**two)/four
      Nxi(18) = (one+xi)*(one-eta)*(one-zeta**two)/four
      Nxi(19) = (one+xi)*(one+eta)*(one-zeta**two)/four
      Nxi(20) = (one-xi)*(one+eta)*(one-zeta**two)/four

      ! the first derivatives: dN/dxi (8x3)
      dNdxi(1,1) = (-(one-eta)*(one-zeta)*(-xi-eta-zeta-two)-     &
                  (one-xi)*(one-eta)*(one-zeta))/eight
      dNdxi(1,2) = (-(one-xi)*(one-zeta)*(-xi-eta-zeta-two)-      &
                  (one-xi)*(one-eta)*(one-zeta))/eight
      dNdxi(1,3) = (-(one-xi)*(one-eta)*(-xi-eta-zeta-two)-       &
                  (one-xi)*(one-eta)*(one-zeta))/eight
      dNdxi(2,1) = ((one-eta)*(one-zeta)*(xi-eta-zeta-two)+       &
                  (one+xi)*(one-eta)*(one-zeta))/eight
      dNdxi(2,2) = (-(one+xi)*(one-zeta)*(xi-eta-zeta-two)-       &
                  (one+xi)*(one-eta)*(one-zeta))/eight
      dNdxi(2,3) = (-(one+xi)*(one-eta)*(xi-eta-zeta-two)-        &
                  (one+xi)*(one-eta)*(one-zeta))/eight
      dNdxi(3,1) = ((one+eta)*(one-zeta)*(xi+eta-zeta-two)+       &
                  (one+xi)*(one+eta)*(one-zeta))/eight
      dNdxi(3,2) = ((one+xi)*(one-zeta)*(xi+eta-zeta-two)+        &
                  (one+xi)*(one+eta)*(one-zeta))/eight
      dNdxi(3,3) = (-(one+xi)*(one+eta)*(xi+eta-zeta-two)-        &
                  (one+xi)*(one+eta)*(one-zeta))/eight
      dNdxi(4,1) = (-(one+eta)*(one-zeta)*(-xi+eta-zeta-two)-     &
                  (one-xi)*(one+eta)*(one-zeta))/eight
      dNdxi(4,2) = ((one-xi)*(one-zeta)*(-xi+eta-zeta-two)+       &
                  (one-xi)*(one+eta)*(one-zeta))/eight
      dNdxi(4,3) = (-(one-xi)*(one+eta)*(-xi+eta-zeta-two)-       &
                (one-xi)*(one+eta)*(one-zeta))/eight
      dNdxi(5,1) = (-(one-eta)*(one+zeta)*(-xi-eta+zeta-two)-     &
                  (one-xi)*(one-eta)*(one+zeta))/eight
      dNdxi(5,2) = (-(one-xi)*(one+zeta)*(-xi-eta+zeta-two)-      &
                  (one-xi)*(one-eta)*(one+zeta))/eight
      dNdxi(5,3) = ((one-xi)*(one-eta)*(-xi-eta+zeta-two)+        &
                  (one-xi)*(one-eta)*(one+zeta))/eight
      dNdxi(6,1) = ((one-eta)*(one+zeta)*(xi-eta+zeta-two)+       &
                  (one+xi)*(one-eta)*(one+zeta))/eight
      dNdxi(6,2) = (-(one+xi)*(one+zeta)*(xi-eta+zeta-two)-       &
                  (one+xi)*(one-eta)*(one+zeta))/eight
      dNdxi(6,3) = ((one+xi)*(one-eta)*(xi-eta+zeta-two)+         &
                  (one+xi)*(one-eta)*(one+zeta))/eight
      dNdxi(7,1) = ((one+eta)*(one+zeta)*(xi+eta+zeta-two)+       &
                  (one+xi)*(one+eta)*(one+zeta))/eight
      dNdxi(7,2) = ((one+xi)*(one+zeta)*(xi+eta+zeta-two)+        &
                  (one+xi)*(one+eta)*(one+zeta))/eight
      dNdxi(7,3) = ((one+xi)*(one+eta)*(xi+eta+zeta-two)+         &
                  (one+xi)*(one+eta)*(one+zeta))/eight
      dNdxi(8,1) = (-(one+eta)*(one+zeta)*(-xi+eta+zeta-two)-     &
                  (one-xi)*(one+eta)*(one+zeta))/eight
      dNdxi(8,2) = ((one-xi)*(one+zeta)*(-xi+eta+zeta-two)+       &
                  (one-xi)*(one+eta)*(one+zeta))/eight
      dNdxi(8,3) = ((one-xi)*(one+eta)*(-xi+eta+zeta-two)+        &
                  (one-xi)*(one+eta)*(one+zeta))/eight
      dNdxi(9,1)  = -two*xi*(one-eta)*(one-zeta)/four
      dNdxi(9,2)  = -(one-xi**two)*(one-zeta)/four
      dNdxi(9,3)  = -(one-xi**two)*(one-eta)/four
      dNdxi(10,1)  = (one-eta**two)*(one-zeta)/four
      dNdxi(10,2)  = -two*eta*(one+xi)*(one-zeta)/four
      dNdxi(10,3)  = -(one-eta**two)*(one+xi)/four
      dNdxi(11,1)  = -two*xi*(one+eta)*(one-zeta)/four
      dNdxi(11,2)  = (one-xi**two)*(one-zeta)/four
      dNdxi(11,3)  = -(one-xi**two)*(one+eta)/four
      dNdxi(12,1)  = -(one-eta**two)*(one-zeta)/four
      dNdxi(12,2)  = -two*eta*(one-xi)*(one-zeta)/four
      dNdxi(12,3)  = -(one-eta**two)*(one-xi)/four
      dNdxi(13,1)  = -two*xi*(one-eta)*(one+zeta)/four
      dNdxi(13,2)  = -(one-xi**two)*(one+zeta)/four
      dNdxi(13,3)  = (one-xi**two)*(one-eta)/four
      dNdxi(14,1)  = (one-eta**two)*(one+zeta)/four
      dNdxi(14,2)  = -two*eta*(one+xi)*(one+zeta)/four
      dNdxi(14,3)  = (one-eta**two)*(one+xi)/four
      dNdxi(15,1)  = -two*xi*(one+eta)*(one+zeta)/four
      dNdxi(15,2)  =  (one-xi**two)*(one+zeta)/four
      dNdxi(15,3)  = (one-xi**two)*(one+eta)/four
      dNdxi(16,1)  = -(one-eta**two)*(one+zeta)/four
      dNdxi(16,2)  = -two*eta*(one-xi)*(one+zeta)/four
      dNdxi(16,3)  = (one-eta**two)*(one-xi)/four
      dNdxi(17,1) = -(one-eta)*(one-zeta**two)/four
      dNdxi(17,2) = -(one-xi)*(one-zeta**two)/four
      dNdxi(17,3) = -zeta*(one-xi)*(one-eta)/two
      dNdxi(18,1) = (one-eta)*(one-zeta**two)/four
      dNdxi(18,2) = -(one+xi)*(one-zeta**two)/four
      dNdxi(18,3) = -zeta*(one+xi)*(one-eta)/two
      dNdxi(19,1) = (one+eta)*(one-zeta**two)/four
      dNdxi(19,2) = (one+xi)*(one-zeta**two)/four
      dNdxi(19,3) = -zeta*(one+xi)*(one+eta)/two
      dNdxi(20,1) = -(one+eta)*(one-zeta**two)/four
      dNdxi(20,2) = (one-xi)*(one-zeta**two)/four
      dNdxi(20,3) = -zeta*(one-xi)*(one+eta)/two
    else
      write(15,*) 'element is not supported for 3D analysis', nNode
      write(*,*)  'element is not supported for 3D analysis', nNode
      return
    endif

    return

  end subroutine interpFunc3

! **********************************************************************

module subroutine faceNodes(nDim,nNode,face,list,nFaceNodes)
  ! this subroutine RETURNs the list of nodes on an
  ! element face for standard 2D and 3D Lagrangian elements
  ! this subroutine is useful for applying traction-type BC

    implicit none

    integer, intent (in) :: nDim, nNode, face
    integer, intent (out):: list(*)
    integer, intent (out):: nFaceNodes

    integer :: list3(3), list4(4)

    if (nDim.eq.2) then
      list3(1:3) = [2,3,1]
      list4(1:4) = [2,3,4,1]

      if (nNode.eq.3) then
        nFaceNodes = 2
        list(1) = face
        list(2) = list3(face)
      else if (nNode.eq.4) then
        nFaceNodes = 2
        list(1) = face
        list(2) = list4(face)
      else if (nNode.eq.6) then
        nFaceNodes = 3
        list(1) = face
        list(2) = list3(face)
        list(3) = face+3
      else if (nNode.eq.8) then
        nFaceNodes = 3
        list(1) = face
        list(2) = list4(face)
        list(3) = face+4
      endif

    else if (nDim.eq.3) then

      if (nNode.eq.4) then
        nFaceNodes = 3
        if (face.eq.1) list(1:3) = [1,2,3]
        if (face.eq.2) list(1:3) = [1,4,2]
        if (face.eq.3) list(1:3) = [2,4,3]
        if (face.eq.4) list(1:3) = [3,4,1]
      else if (nNode .eq.6) then
        nFaceNodes = 3
        if (face.eq.1) list(1:3) = [1,2,3]
        if (face.eq.2) list(1:3) = [6,5,4]
        if (face.eq.3) list(1:4) = [1,2,5,4]
        if (face.eq.4) list(1:4) = [2,3,6,5]
        if (face.eq.5) list(1:4) = [4,6,3,1]
        if (face>2) nFaceNodes = 4
      else if (nNode.eq.10) then
        nFaceNodes = 6
        if (face.eq.1) list(1:6) = [1,2,3,5,6,7]
        if (face.eq.2) list(1:6) = [1,4,2,8,9,5]
        if (face.eq.3) list(1:6) = [2,4,3,9,10,6]
        if (face.eq.4) list(1:6) = [3,4,1,10,8,7]
      else if (nNode.eq.8) then
        nFaceNodes = 4
        if (face.eq.1) list(1:4) = [1,2,3,4]
        if (face.eq.2) list(1:4) = [5,8,7,6]
        if (face.eq.3) list(1:4) = [1,5,6,2]
        if (face.eq.4) list(1:4) = [2,6,7,3]
        if (face.eq.5) list(1:4) = [3,7,8,4]
        if (face.eq.6) list(1:4) = [4,8,5,1]
      else  if (nNode.eq.20) then
        nFaceNodes = 8
        if (face.eq.1) list(1:8) = [1,2,3,4,9,10,11,12]
        if (face.eq.2) list(1:8) = [5,8,7,6,16,15,14,13]
        if (face.eq.3) list(1:8) = [1,5,6,2,17,13,18,9]
        if (face.eq.4) list(1:8) = [2,6,7,3,18,14,19,10]
        if (face.eq.5) list(1:8) = [3,7,8,4,19,15,6,11]
        if (face.eq.6) list(1:8) = [4,8,5,1,20,16,17,12]
      endif
    endif

    return

  end subroutine faceNodes

end submodule interpolation

! **********************************************************************
! **********************************************************************


! **********************************************************************
! ******************* GAUSSIAN QUADRATURE SUBMODULE ********************
! **********************************************************************
! integration schemes:  (a) reduced and full integration: bar elements
!                       (b) full integration: tri and tet elements
!                       (c) reduced and full integration: quad and hex elements
! **********************************************************************
! **********************************************************************

submodule (lagarange_elem) quadrature

  contains

! **********************************************************************

module subroutine gaussQuadrtr1(nNode,nInt,w,xi)

    use global_parameters
    implicit none

    integer:: nNode, nInt
    double precision:: w(nInt), xi(nInt,1)

    w = zero
    xi = zero

    if (nNode.eq.2) then
      if (nInt.eq.1) then       ! full integration for bar2
        w = zero
        xi = two
      else
        write(15,*) 'wrong Gauss points for bar2 element', nInt
        write(*,*)  'wrong Gauss points for bar2 element', nInt
        return
      endif       ! end int for 2-noded bar

    elseif (nNode.eq.3) then
      if (nInt .eq. 2) then     ! reduced integration for bar3
        w(1:2)  = one
        xi(1,1) = -dsqrt(third)
        xi(2,1) = dsqrt(third)
      elseif (nInt.eq.3) then   ! full integration for bar3
        w(1) = five/nine
        w(2) = eight/nine
        w(3) = five/nine
        xi(1,1) = -dsqrt(three/five)
        xi(2,1) = zero
        xi(3,1) = dsqrt(three/five)
      else
        write(15,*) 'wrong Gauss points for bar3 element', nInt
        write(*,*)  'wrong Gauss points for bar3 element', nInt
        return
      endif         ! end int for 3-noded bar

    else
      write(15,*) 'elements not supported for 1D analysis', nNode
      write(*,*)  'elements not supported for 1D analysis', nNode
      return
    endif

    return

  end subroutine gaussQuadrtr1

  ! **********************************************************************

module subroutine gaussQuadrtr2(nNode,nInt,w,xi)

    ! this subroutines returns the weights and
    ! gauss point coordinate of 2D Lagrangian elements
    ! currently supports: nInt = 1 (tri3) and 3 (tri6)
    !                     nInt = 1, 4 (quad4) and 4, 9 (quad8)

    use global_parameters
    implicit none

    integer:: nNode, nInt
    double precision:: x1D(4), w1D(4)
    double precision:: w(nInt), xi(nInt,2)

    w  = zero
    xi = zero

    if (nNode.eq.3) then      ! plane tri3 elements (full integration)
      if (nInt.eq.1) then
        w(1) = half
        xi(1,1) = third
        xi(2,1) = third
      else
        write(15,*) 'wrong Gauss points for tri3 element', nInt
        write(*,*)  'wrong Gauss points for tri3 element', nInt
        return
      endif

    elseif (nNode.eq.6) then  ! plane tri6 elements (full integration)
      if (nInt.eq.3) then
        w(1:3) = sixth

        xi(1,1) = half
        xi(1,2) = half
        xi(2,1) = zero
        xi(2,2) = half
        xi(3,1) = half
        xi(3,2) = zero
      else
        write(15,*) 'wrong Gauss points for tri6 element', nInt
        write(*,*)  'wrong Gauss points for tri6 element', nInt
        return
      endif

    elseif((nNode.eq.4)) then ! plane quad4 element

      if (nInt.eq.1) then     ! reduced integration for quad4
        w = four

        xi(1,1) = zero
        xi(1,2) = zero

      elseif (nInt.eq.4) then ! full integration for quad4

        w(1:4) = one

        x1D(1) = dsqrt(third)
        xi(1,1) = -x1D(1)
        xi(1,2) = -x1D(1)
        xi(2,1) = x1D(1)
        xi(2,2) = -x1D(1)
        xi(3,1) = -x1D(1)
        xi(3,2) = x1D(1)
        xi(4,1) = x1D(1)
        xi(4,2) = x1D(1)
      else
        write(15,*) 'wrong Gauss points for quad4 element', nInt
        write(*,*)  'wrong Gauss points for quad4 element', nInt
        return
      endif

    elseif (nNode.eq.8) then  ! plane quad8 element

      if (nInt.eq.4) then     ! reduced integration for quad8

        w(1:4) = one

        x1D(1) = dsqrt(third)
        xi(1,1) = -x1D(1)
        xi(1,2) = -x1D(1)
        xi(2,1) = x1D(1)
        xi(2,2) = -x1D(1)
        xi(3,1) = -x1D(1)
        xi(3,2) = x1D(1)
        xi(4,1) = x1D(1)
        xi(4,2) = x1D(1)

      elseif(nInt.eq.9) then  ! full integration for quad8
        w1D(1) = (five/nine)*(five/nine)
        w1D(2) = (five/nine)*(eight/nine)
        w1D(3) = (eight/nine)*(eight/nine)

        w(1) = w1D(1)
        w(2) = w1D(2)
        w(3) = w1D(1)
        w(4) = w1D(2)
        w(5) = w1D(3)
        w(6) = w1D(2)
        w(7) = w1D(1)
        w(8) = w1D(2)
        w(9) = w1D(1)

        x1D(1) = dsqrt(three/five)
        xi(1,1) = -x1D(1)
        xi(1,2) = -x1D(1)
        xi(2,1) = zero
        xi(2,2) = -x1D(1)
        xi(3,1) = x1D(1)
        xi(3,2) = -x1D(1)
        xi(4,1) = -x1D(1)
        xi(4,2) = zero
        xi(5,1) = zero
        xi(5,2) = zero
        xi(6,1) = x1D(1)
        xi(6,2) = zero
        xi(7,1) = -x1D(1)
        xi(7,2) = x1D(1)
        xi(8,1) = zero
        xi(8,2) = x1D(1)
        xi(9,1) = x1D(1)
        xi(9,2) = x1D(1)

      else
        write(15,*) 'wrong Gauss points for quad8 element', nInt
        write(*,*)  'wrong Gauss points for quad8 element', nInt
        return
      endif

    else
      write(15,*) 'elements not supported for 2D analysis', nNode
      write(*,*)  'elements not supported for 2D analysis', nNode
      return
    endif


    return

  end subroutine gaussQuadrtr2

  ! **********************************************************************

module subroutine gaussQuadrtr3(nNode,nInt,w,xi)

    ! this subroutines returns the weights and
    ! gauss point coordinate of 3D Lagrangian elements
    ! currently supports: nInt = 1 (tet4) and 4 (tet10)
    !                     nInt = 1, 8 (hex8) and 8, 27 (hex20)

    use global_parameters
    implicit none

    integer:: nNode, nInt
    integer:: i, j, k, n
    double precision:: x1D(4), w1D(4)
    double precision:: w(nInt), xi(nInt,3)

    w  = zero
    xi = zero

    if(nNode.eq.4) then       ! 3D tet4 element (full integration)
      if(nInt.eq.1) then
        w(1) = sixth
        xi(1,1:3) = fourth

      else
        write(15,*) 'wrong Gauss points for tet4 element', nInt
        write(*,*)  'wrong Gauss points for tet4 element', nInt
        return
      endif

    elseif(nNode.eq.10) then  ! 3D tet10 element (full integration)

      if (nInt.eq.4) then
        w(1:4) = one/24.0d0

        x1D(1) = 0.58541020d0
        x1D(2) = 0.13819660d0

        xi(1,1) = x1D(1)
        xi(2,1) = x1D(2)
        xi(3,1) = x1D(2)
        xi(1,2) = x1D(2)
        xi(2,2) = x1D(1)
        xi(3,2) = x1D(2)
        xi(1,3) = x1D(2)
        xi(2,3) = x1D(2)
        xi(3,3) = x1D(1)
        xi(4,1) = x1D(2)
        xi(4,2) = x1D(2)
        xi(4,3) = x1D(2)

      else
        write(15,*) 'wrong Gauss points for tet10 element', nInt
        write(*,*) 'wrong Gauss points for tet10 element', nInt
        return
      endif

    elseif(nNode.eq.8) then   ! 3D hex8 element

      if(nInt.eq.1) then      ! reduced integration for hex8
        w(1) = eight
        xi(1,1:3) = zero

      elseif(nInt.eq.8) then  ! full-integration for hex8
        w(1:8) = one

        x1D(1) = -dsqrt(third)
        x1D(2) = dsqrt(third)

        do k = 1,2
          do j = 1,2
            do i = 1,2
              n = 4*(k-1) + 2*(j-1) + i
              xi(n,1) = x1D(i)
              xi(n,2) = x1D(j)
              xi(n,3) = x1D(k)
            end do
          end do
        end do

      else
        write(15,*) 'wrong Gauss points for hex8 element', nInt
        write(*,*)  'wrong Gauss points for hex8 element', nInt
        return
      endif

    elseif(nNode.eq.20) then  ! 3D hex20 element

      if (nInt.eq.8) then     ! reduced integration for hex20
        w(1:8) = one

        x1D(1) = -dsqrt(third)
        x1D(2) = dsqrt(third)

        do k = 1,2
          do j = 1,2
            do i = 1,2
              n = 4*(k-1) + 2*(j-1) + i
              xi(n,1) = x1D(i)
              xi(n,2) = x1D(j)
              xi(n,3) = x1D(k)
            end do
          end do
        end do

      elseif(nInt.eq.27) then ! full integration for hex20
        w1D(1) = five/nine
        w1D(2) = eight/nine
        w1D(3) = w1D(1)

        x1D(1) = -dsqrt(0.6d0)
        x1D(2) = zero
        x1D(3) = dsqrt(0.6d0)
        do k = 1,3
          do j = 1,3
            do i = 1,3
              n = 9*(k-1) + 3*(j-1) + i
              xi(n,1) = x1D(i)
              xi(n,2) = x1D(j)
              xi(n,3) = x1D(k)
              w(n) = w1D(i)*w1D(j)*w1D(k)
            end do
          end do
        end do

      else
        write(15,*) 'wrong Gauss points for hex20 element', nInt
        write(*,*)  'wrong Gauss points for hex20 element', nInt
        return
      endif

    else
      write(15,*) 'element is not supported for 3D analysis', nNode
      write(*,*)  'element is not supported for 3D analysis', nNode
      return
    endif

    return

  end subroutine gaussQuadrtr3

end submodule quadrature

  ! **********************************************************************
