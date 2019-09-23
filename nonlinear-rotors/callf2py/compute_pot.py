import pot
#print(pot.__doc__)
com1=[0.0,0.0,0.0]
com2=[0.0,0.0,10.0]
Eulang2=[-2.0,1.0,1.0]
Eulang1=[2.0,1.0,1.0]
c=pot.caleng(com1,com2,Eulang1,Eulang2)
print(c)
