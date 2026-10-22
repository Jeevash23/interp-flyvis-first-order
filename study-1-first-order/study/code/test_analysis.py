"""Behavioral checks for scientific scoring edge cases."""
import unittest
import numpy as np
from analyse_revision import top_budget, correlation, bounded_prediction, path_sums
from revision_common import independent_metrics, causal_frames, stimulus, legacy_panel, load_npz


class ScientificScoringTests(unittest.TestCase):
    def test_constant_predictions_have_random_screening_and_undefined_correlation(self):
        truth=np.array([True,True,False,False,False])
        result=top_budget(np.zeros(5),truth,2)
        self.assertAlmostEqual(result["precision"],2/5)
        self.assertAlmostEqual(result["recall"],2/5)
        self.assertIsNone(correlation(np.zeros(5),np.arange(5)))

    def test_boundary_ties_have_fractional_credit(self):
        result=top_budget(np.array([4,3,3,1]),np.array([False,True,False,True]),2)
        self.assertAlmostEqual(result["precision"],0.25)
        self.assertAlmostEqual(result["recall"],0.25)

    def test_projection_does_not_invent_improvement_beyond_bounds(self):
        base=0.7
        truth=np.linspace(0,1,31)-base
        prediction=np.linspace(-3,3,31)
        clipped=bounded_prediction(prediction,base)
        self.assertTrue(np.all(np.abs(clipped-truth)<=np.abs(prediction-truth)+1e-14))

    def test_indirect_paths_are_detected_without_inventing_direct_edges(self):
        adjacency=np.array([[0,0,0],[2,0,0],[0,3,0]],float)
        score=path_sums(adjacency,2)
        self.assertEqual(score[1][2,0],0)
        self.assertEqual(score[2][2,0],1)

    def test_causal_prefix_preserves_original_metrics(self):
        for key in ["dev_fast","final_slow"]:
            model="000" if key=="dev_fast" else "001"
            old=load_npz(legacy_panel(model,key)/"baseline.npz")
            measured=independent_metrics(old["trace"][:,:causal_frames(key)],stimulus(key))
            for field in ("dsi","peaks","raw_mean","raw_integral","denominator"):
                np.testing.assert_allclose(measured[field],old[field],atol=2e-6,rtol=2e-5)

    def test_abolition_has_no_directional_information(self):
        stim=stimulus("dev_fast")
        trace=np.zeros((24,causal_frames("dev_fast"),8),np.float32)
        measured=independent_metrics(trace,stim)
        self.assertTrue(np.all(measured["denominator"]==0))
        # A computational zero is never sufficient for scientific eligibility.
        self.assertFalse(np.any(measured["denominator"]>1e-8))


if __name__=="__main__":
    unittest.main()
