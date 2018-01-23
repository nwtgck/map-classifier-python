import unittest

import map_classifier

class MapClassifierTest(unittest.TestCase):

  def test_prediction(self):
    self.assertTrue(True)


def suite():
  suite = unittest.TestSuite()
  suite.addTest(unittest.makeSuite(MapClassifierTest))
  return suite
