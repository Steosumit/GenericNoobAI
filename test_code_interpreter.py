import unittest
from code_interpreter import CodeInterpreter


class TestCodeInterpreter(unittest.TestCase):
    def setUp(self):
        self.interpreter = CodeInterpreter()

    def test_simple_print(self):
        """Test basic print statement execution"""
        code = "print('Hello, World!')"
        result = self.interpreter.execute_code(code)
        self.assertIn("Hello, World!", str(result))

    def test_arithmetic_operations(self):
        """Test arithmetic calculations"""
        code = """
result = 10 + 5 * 2
print(result)
"""
        result = self.interpreter.execute_code(code)
        self.assertIn("20", str(result))

    def test_variable_assignment(self):
        """Test variable assignment and usage"""
        code = """
x = 42
y = x * 2
print(f"Result: {y}")
"""
        result = self.interpreter.execute_code(code)
        self.assertIn("Result: 84", str(result))

    def test_error_handling(self):
        """Test that errors are properly caught and reported"""
        code = "print(undefined_variable)"
        result = self.interpreter.execute_code(code)
        self.assertIn("Error", str(result))

    def test_multi_line_code(self):
        """Test execution of multi-line code"""
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(7))
"""
        result = self.interpreter.execute_code(code)
        self.assertIn("13", str(result))

    def test_complete_code_block(self):
        """Test a complete Python code block with imports, functions, and logic"""
        code = """
import math

# Calculate area of circle
def calculate_area(radius):
    return math.pi * radius ** 2

# Test with different radii
radii = [1, 2, 3, 5]
for r in radii:
    area = calculate_area(r)
    print(f"Circle with radius {r}: area = {area:.2f}")

# Calculate sum of squares
numbers = [1, 2, 3, 4, 5]
sum_of_squares = sum([n**2 for n in numbers])
print(f"Sum of squares: {sum_of_squares}")
"""
        result = self.interpreter.execute_code(code)
        self.assertIn("Circle with radius", str(result))
        self.assertIn("Sum of squares: 55", str(result))


if __name__ == '__main__':
    unittest.main()
