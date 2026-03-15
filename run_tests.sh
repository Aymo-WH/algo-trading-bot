export PYTHONPATH=$PYTHONPATH:.
for test_file in tests/test_*.py; do
  echo "Running $test_file"
  python "$test_file"
done
