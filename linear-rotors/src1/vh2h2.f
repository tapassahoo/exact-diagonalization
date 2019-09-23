c ----------------------------------------------------------------------

c     This subroutine computes the H2-H2 potential energy surface as
c     described in the Journal of Chemical Physics article "A six-
c     dimensional H2-H2 potential energy surface for bound state
c     spectroscopy" by Robert J. Hinde.

c     Please direct comments to rhinde@utk.edu or hinde.robert@gmail.com

c     This version of the subroutine is dated 1 June 2010.  An earlier
c     version of the subroutine (dated 31 October 2007) contained an
c     error in the computation of the the potential energy at fixed
c     (r, t1, t2, phi) as a power series in (r1, r2).  Specifically,
c     the term including the coefficient c20 contained a typographical
c     error, and was given as c20*(r2-r0)**2.  This has been changed
c     to the correct expression c20*(r1-r0)**2.  Piotr Jankowski first
c     notified me of this problem and I appreciate his assistance in
c     locating the typographical error.

c     Before calling this subroutine, your program must call the vinit
c     subroutine that performs some initialization work.  The variables
c     used in this initialization work are stored in the common block
c     /vh2h2c/.

c     The subroutine takes six input parameters:

c     r = H2-H2 distance in bohr
c     r1 = bond length of monomer 1 in bohr
c     r2 = bond length of monomer 2 in bohr
c     t1 = theta1 in radians
c     t2 = theta2 in radians
c     phi = phi in radians

c     The potential energy is returned to the calling program in the
c     variable potl, in units of wavenumbers.

      subroutine vh2h2(r, r1, r2, t1, t2, phi, potE)

      implicit real*8 (a-h, o-z)

      dimension v(4), vv(9), vij(3, 3, 4),potl(8)

      common /vh2h2c/ coef(18, 3, 3, 4, 4),
     +                arep(3, 3, 4), crep(3, 3, 4),
     +                c5(3, 3), c6(3, 3, 4), c8(3, 3, 4), cten(3, 3)

      parameter (r0=1.4d0)
      parameter (dr=0.3d0)

c --- test for R outside spline endpoints.

      if (r.le.4.210625d0) then

c         write (6, 6009)
c6009     format ('warning 6009 in VH2H2: R smaller than recommended')

         do i=1, 3
         do j=1, 3
         do k=1, 4
            vij(i, j, k)=arep(i, j, k)*exp(r*crep(i, j, k))
         end do
         end do
         end do

         goto 200

      else if (r.ge.12.0d0) then

         do i=1, 3
         do j=1, 3
            vij(i, j, 1)=cten(i, j)/r**10
            vij(i, j, 4)=c5(i, j)/r**5
            do k=2, 3
               vij(i, j, k)=0.0d0
            end do
         end do
         end do

         do i=1, 3
         do j=1, 3
         do k=1, 4
            vij(i, j, k)=vij(i, j, k)+c6(i, j, k)/r**6+c8(i, j, k)/r**8
         end do
         end do
         end do

         goto 200

      end if

c --- compute "old R" before shift.

      rold=(r+6.5d0*0.0175d0)/1.0175d0

c --- compute index into spline look-up tables.

      if (rold.lt.6.5d0) then

c ------ for R < 6.5, it's more convenient to use the old R.

         idx=idint((rold-4.25d0)*4.0d0)+1

      else if (r.ge.6.5d0.and.r.lt.10.0d0) then

         idx=idint((rold-6.5d0)*2.0d0)+10

      else if (r.ge.10.0d0.and.r.lt.12.0d0) then

         idx=idint(rold-10.0d0)+17

      else

         write (6, 6001)
6001     format ('error 6001 in VH2H2: spline look-up failed!')

         goto 999

      end if

c --- evaluate splines for all 9 (r1, r2) pairs.

      do i=1, 3
      do j=1, 3
      do k=1, 4
         vij(i, j, k)=0.0d0
      end do
      end do
      end do

      do i=1, 3
      do j=1, 3
      do k=1, 4
      do l=1, 4
         vij(i, j, k)=r*vij(i, j, k)+coef(idx, i, j, k, l)
      end do
      end do
      end do
      end do

c --- use power series in (r1, r2) to compute potential energy.

200   do k=1, 4

         do i=1, 3
         do j=1, 3
            vv(3*(i-1)+j)=vij(i, j, k)
         end do
         end do


c ------ perform symmetry checks.

         if (k.eq.1.or.k.eq.4) then

            if (vv(2).ne.vv(4)) then
               write (6, 6002) 2, 4, k
               goto 999
            else if (vv(3).ne.vv(7)) then
               write (6, 6002) 3, 7, k
               goto 999
            else if (vv(6).ne.vv(8)) then
               write (6, 6002) 6, 8, k
               goto 999
            end if

         end if

6002     format ('error 6002 in VH2H2: vv(', i1, ') and vv(', i1, ') ',
     +           'do not match for angular term ', i1)

c ------ compute power series coefficients.

         c00=vv(5)

         c01=0.5d0*(vv(6)-vv(4))/dr
         c10=0.5d0*(vv(8)-vv(2))/dr

         c02=-vv(5)/(dr**2)+0.5d0*(vv(6)+vv(4))/(dr**2)
         c20=-vv(5)/(dr**2)+0.5d0*(vv(8)+vv(2))/(dr**2)

         c11=0.25d0*(vv(9)-vv(3)-(vv(7)-vv(1)))/(dr**2)

         aa=0.5d0*(vv(3)+vv(1)-2.0d0*vv(2))/(dr**2)
         bb=0.5d0*(vv(9)+vv(7)-2.0d0*vv(8))/(dr**2)

         c12=0.5d0*(bb-aa)/dr

         aa=0.5d0*(vv(7)+vv(1)-2.0d0*vv(4))/(dr**2)
         bb=0.5d0*(vv(9)+vv(3)-2.0d0*vv(6))/(dr**2)

         c21=0.5d0*(bb-aa)/dr

         cc=0.5d0*(vv(8)+vv(2)-2.0d0*vv(5))/(dr**2)

         c22=0.5d0*(aa+bb-2.0d0*cc)/(dr**2)

c ------ evaluate power series.

         v(k)=c00+
     +        c01*(r2-r0)+
     +        c10*(r1-r0)+
     +        c02*(r2-r0)**2+
     +        c20*(r1-r0)**2+
     +        c11*(r1-r0)*(r2-r0)+
     +        c12*(r1-r0)*(r2-r0)**2+
     +        c21*(r1-r0)**2*(r2-r0)+
     +        c22*((r1-r0)*(r2-r0))**2

      end do

c --- compute the angular functions.

      c1=cos(t1)
      c2=cos(t2)
      s1=sin(t1)
      s2=sin(t2)

      g000=1.0d0
      g202=2.5d0*(3.0d0*c1**2-1.0d0)
      g022=2.5d0*(3.0d0*c2**2-1.0d0)
      g224=45.0d0/(4.0d0*sqrt(70.0d0))*
     *     (0.32d0*g022*g202-16.0d0*s1*c1*s2*c2*cos(phi)+
     +     (s1*s2)**2*cos(2.0d0*phi))

c --- compute the potential energy from the angular functions and the
c     interpolated (r1, r2)-dependent coefficients.

      potE=v(1)*g000+v(2)*g202+v(3)*g022+v(4)*g224
      potl(1)=v(1)
      potl(2)=v(2)
      potl(3)=v(3)
      potl(4)=v(4)
      potl(5)=g000
      potl(6)=g202
      potl(7)=g022
      potl(8)=g224

c      print *, potE

c --- this write statement is for debugging purposes.

c     write (6, 6123) v(1), g000, v(1)*g000,
c    +                v(2), g202, v(2)*g202,
c    +                v(3), g022, v(3)*g022,
c    +                v(4), g224, v(4)*g224, potl
6123  format ('coefficients and angular functions:'/,
     +   'A000 = ', f12.7, ' g000 = ', f12.7, ' product = ', f12.7/,
     +   'A202 = ', f12.7, ' g202 = ', f12.7, ' product = ', f12.7/,
     +   'A022 = ', f12.7, ' g022 = ', f12.7, ' product = ', f12.7/,
     +   'A224 = ', f12.7, ' g224 = ', f12.7, ' product = ', f12.7/,
     +   50x, '------------'/, 44x, 'sum = ', f12.7/)

      return

999   write (6, 6999) r, r1, r2, t1, t2, phi
6999  format ('VH2H2 input parameters:'/,
     +        'R = ', 1pe15.8/,
     +        'r1, r2 = ', 1pe15.8, 1x, 1pe15.8/,
     +        't1, t2 = ', 1pe15.8, 1x, 1pe15.8/,
     +        'phi = ', 1pe15.8)

      stop
      end

c ----------------------------------------------------------------------

c     This is the initialization subroutine that must be called once
c     before using the preceding subroutine.

c     It reads data from three files:

c     all_coefs
c     short_range
c     long_range

      subroutine vinit

      implicit real*8 (a-h, o-z)

      common /vh2h2c/ coef(18, 3, 3, 4, 4),
     +                arep(3, 3, 4), crep(3, 3, 4),
     +                c5(3, 3), c6(3, 3, 4), c8(3, 3, 4), cten(3, 3)

      open (3, file='all_coefs')

      do i=1, 3
      do j=1, 3
      do k=1, 4
      do n=1, 18
         read (3, *) (coef(n, i, j, k, l), l=1, 4)
      end do
      end do
      end do
      end do

      close (3)

      open (3, file='short_range')

      do i=1, 3
      do j=1, 3
      do k=1, 4
         read (3, *) arep(i, j, k), crep(i, j, k)
      end do
      end do
      end do

      close (3)

      open (3, file='long_range')

      do i=1, 3
      do j=1, 3

         read (3, *) c5(i, j)

         read (3, *) c6(i, j, 1), c8(i, j, 1), cten(i, j)

         do k=2, 4
            read (3, *) c6(i, j, k), c8(i, j, k)
         end do

      end do
      end do

      return
      end

