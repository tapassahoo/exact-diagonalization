c     ==========================================================
      subroutine cluster(com_1,com_2,Eulang_1,Eulang_2,E_12)
c     ==========================================================
c     Initially:
c     __________________________________________________
c     this subroutine calculates pair potential between
c     rigid HF molecules encapsulated in C60 given the
c     coordinates of centre of mass of HF molecules and
c     the Euler angles describing the rotation of each HF
c     input: com_1, com_2, Eulang_1, Eulang_2 
c     output: E_12
c     phi=Eulang(1)
c     theta=Eulang(2)
c     chi=Eulang(3) for HF chi = 0
c     __________________________________________________
c     rotmat, computed within the code with Eulang
c     ROwf, RH1wf, RH2wf, RCwf put as data   
      implicit double precision(a-h,o-z)
      dimension delta(3,3),  
     +          com_1(3), com_2(3),d(3),d1(3),d2(3),
     +          Eulang_1(3),Eulang_2(3),R12(3),
     +          rotmat_1(3,3),rotmat_2(3,3),
     +          ROwf(3),RH1wf(3),RH2wf(3),
     +          RO_1_sf(3),RO_2_sf(3),
     +          RH1_1_sf(3),RH1_2_sf(3),
     +          RH2_1_sf(3),RH2_2_sf(3),
     +          alfa_c60(3,3),alfa_c60_1(3,3),alfa_c60_2(3,3),
     +          alfa_h2oc60(3,3),alfa_h2oc60_1(3,3),alfa_h2oc60_2(3,3),
     +          quad(3,3),q1(3,3),q2(3,3),
     +          om(3,3,3),om1(3,3,3),om2(3,3,3),
     +          F(3,3,3,3),F1(3,3,3,3),F2(3,3,3,3)
      parameter (kcal2k=503.218978939d0,polautoA3=0.14819d0,
     +          quadtoDebA=(2.54177d0*0.529177d0),a0=0.529177d0,
     +          am0=2.54175d0)
      autocm=219474.63d0 !from hartree to cm-1
      autoK=315775.13d0 !hartree to Kelvin
      autokcal=627.503d0!hartree to kcal/mol
      autokJ=2625.5d0!hartree to kJ/mol
c permanent dipole moment of hfc60  
      data d/0.00d0,0.0d0,0.45d0/ !in ea0 pbe0/def2-tzvpp 



c delta function     
      do i=1,3
        do j=1,3
          if (i.eq.j) then
            delta(i,j)=1.0d0
          else
            delta(i,j)=0.0d0
          endif
        enddo
      enddo

      E_12=0.d0

c     prepare rotational matrix for water1 
c     and coordinates of atoms in space sixed frame
      call matpre(Eulang_1, rotmat_1)
  

c     prepare rotational matrix for water2
c     and coordinates of atoms in space sixed frame
      call matpre(Eulang_2, rotmat_2)

c dipole moment of each water@c60 molecule in space fixed frame
      d1(1:3)=0.d0
      d2(1:3)=0.d0
      do i=1,3
       do j=1,3
         d1(i)=d1(i)+rotmat_1(i,j)*d(j)
         d2(i)=d2(i)+rotmat_2(i,j)*d(j)
       enddo
      enddo

      R12(1:3)=0.0
      do i=1,3
         R12(i)=com_1(i)-com_2(i)
      enddo
      R=dsqrt(R12(1)*R12(1)+R12(2)*R12(2)+R12(3)*R12(3))

c ... pair interaction potential
c dipole-dipole interaction
      v3_elec=0.d0
      do i=1,3
      	do j=1,3
          v3_elec = v3_elec - d1(i)*d2(j)*R**(-5.d0)*(3.d0*R12(i)*R12(j)
     &            - R**2.d0*delta(i,j))   
        enddo
      enddo
      

      v3_elec=v3_elec*a0**3.d0  


      E_12_au=v3_elec !in Hartree
      E_12=E_12_au*autoK !in Kelvin
      E_12_kcal=E_12_au*autokcal !in Kcal/mol  
      E_12_kJ=E_12_au*autokJ !in kJ/mol
      E_12_cm=E_12_au*autocm !in cm-1      
 
c      write(*,*) E_12
c      write(44,72) RO_1_sf,RH1_1_sf,RH2_1_sf      
c      write(44,73) RO_2_sf,RH1_2_sf,RH2_2_sf
c      write(44,71) com_1(1:3),com_2(1:3),R12(1:3),R
c      write(44,74) E_12,E_12_cm
c      write(44,*)  "            "
   71 format("R1=(",3F8.4,")   R2=(",3F8.4,")   R12=(",3F8.4,")  R=",
     1      1F8.4, "   Angstrom")
   72 format("1st mol  O=(",3F12.8,")  H=(",3F12.8,")  H=(",3F12.8,")")
   73 format("2nd mol  O=(",3F12.8,")  H=(",3F12.8,")  H=(",3F12.8,")")
   74 format("EKel = ",1F10.4,"  Ecm-1 = ",1F10.4)
      return
      end

      subroutine matpre(Eulang,rotmat)
      implicit double precision(a-h,o-z)

      dimension Eulang(3),rotmat(3,3)

      phi=Eulang(1)
      theta=Eulang(2)
      chi=Eulang(3)

      cp=cos(phi)
      sp=sin(phi)
      ct=cos(theta)
      st=sin(theta)
      ck=cos(chi)
      sk=sin(chi)

      rotmat(1,1)=cp*ct*ck-sp*sk
      rotmat(1,2)=-cp*ct*sk-sp*ck
      rotmat(1,3)=cp*st
      rotmat(2,1)=sp*ct*ck+cp*sk
      rotmat(2,2)=-sp*ct*sk+cp*ck
      rotmat(2,3)=sp*st
      rotmat(3,1)=-st*ck
      rotmat(3,2)=st*sk
      rotmat(3,3)=ct

      return
      end
