from loops import Loops

import unittest
import subprocess

class TestLoops(unittest.TestCase):
    def test_read_loops(self):
        loop_file = 'juicer/data/generated_data/loops/merged_loops_r_500_1000_2000.bedpe'
        loops = Loops(loop_file)
        df = loops._read_loops()
        self.assertCountEqual(df.columns.tolist(), ['start', 'end', 'res'])
        
        # Count number of lines in bedpe file
        s = subprocess.check_output(["wc", "-l", loop_file])
        
        self.assertEqual(len(df), int(s.split()[0]) - 2)

if __name__ == '__main__':
    unittest.main()