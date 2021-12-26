import pot
import qpot
import ggpot
import qspc
#print(pot.__doc__)
com1=[0.0,0.0,0.0]
com2=[0.0,0.0,10.0]
Eulang2=[0.0, -1.0, 0.0]
Eulang1=[0.0, 1.0, 0.0]
c=pot.caleng(com1,com2,Eulang1,Eulang2)
d=qpot.caleng(com1,com2,Eulang1,Eulang2)
e=ggpot.caleng(com1,com2,Eulang1,Eulang2)
f=qspc.caleng(com1,com2,Eulang1,Eulang2)
print("tip4p V = "+str(c))
print("qtip4p V = "+str(d))
print("ggpot V = "+str(e))
print("qspc V = "+str(f))
