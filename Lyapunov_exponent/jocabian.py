import sympy

a, b, x, y = sympy.symbols("a b x y")
a = 2 * x ** 3 - y ** 2 - 1
b = x * y ** 3 - y - 4
funcs = sympy.Matrix([a, b])
args = sympy.Matrix([x, y])
res = funcs.jacobian(args)
print(res)
