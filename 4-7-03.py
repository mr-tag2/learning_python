import numpy as np


print(np.version)

print('--------------------')

p1=[3,8,10.1]

q=np.array(p1,dtype='int8')

print(q.ndim)
print(q.size)
print(q.itemsize)
print(q)

print('--------------------')

q1= np.zeros((3,5))
print(q1)

print('--------------------')

q2= np.ones((3,5))
print(q2)


print('--------------------')

q3= np.eye(5)
print(q3)

print('--------------------')

q4= range(10)
print(q4)


print('--------------------')

q5= np.linspace(1, 6,10)
print(q5)

print('--------------------')

q5= np.full((3, 4),'d')
print(q5)

np.save('d://Prj//_leaning//py',q5)

p=np.load('d://Prj//_leaning//py')

print(p)
