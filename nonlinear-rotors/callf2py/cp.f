C
      SUBROUTINE ZADD(A,B,C,N)
C
CF2PY INTENT(OUT) :: C
CF2PY INTENT(HIDE) :: N
CF2PY REAL* :: A(N)
CF2PY REAL*8 :: B(N)
CF2PY REAL*8 :: C(N)
      REAL*8 A(*)
      REAL*8 B(*)
      REAL*8 C(*)
      INTEGER N
      CALL ZADD1(A,B,N)
      DO 20 J = 1, N
         C(J) = A(J) + B(J)
 20   CONTINUE
      END

      SUBROUTINE ZADD1(A,B,N)
C
CF2PY INTENT(IN,OUT) :: A
CF2PY INTENT(IN,OUT) :: B
CF2PY INTENT(HIDE) :: N
CF2PY REAL*8 :: A(N)
CF2PY REAL*8 :: B(N)
      REAL*8 A(*)
      REAL*8 B(*)
      INTEGER N
      DO 20 J = 1, N
         A(J) = A(J)*2.0
         B(J) = B(J)*2.0
 20   CONTINUE
      END