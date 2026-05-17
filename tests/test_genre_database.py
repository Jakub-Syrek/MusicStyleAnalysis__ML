"""Unit tests for the static genre profile database."""

import unittest

from src.genre_database import GENRE_DATABASE


REQUIRED_KEYS = {
    "tempo_range",
    "loudness_range",
    "spectral_centroid_range",
    "zcr_range",
    "description",
    "family",
}


class TestGenreDatabaseSchema(unittest.TestCase):
    """Validate the structural invariants of ``GENRE_DATABASE``."""

    def test_database_is_not_empty(self) -> None:
        """The shipped database must define at least one genre profile."""
        self.assertGreater(len(GENRE_DATABASE), 0)

    def test_each_profile_has_required_keys(self) -> None:
        """Every profile must expose tempo/loudness/spectral/zcr/family/description."""
        for genre, profile in GENRE_DATABASE.items():
            missing = REQUIRED_KEYS - set(profile.keys())
            self.assertFalse(
                missing,
                msg=f"Genre '{genre}' is missing keys: {missing}",
            )

    def test_ranges_are_valid_tuples(self) -> None:
        """Range fields must be ``(low, high)`` tuples with ``low <= high``."""
        range_fields = (
            "tempo_range",
            "loudness_range",
            "spectral_centroid_range",
            "zcr_range",
        )
        for genre, profile in GENRE_DATABASE.items():
            for field in range_fields:
                low, high = profile[field]
                self.assertIsInstance(low, (int, float))
                self.assertIsInstance(high, (int, float))
                self.assertLessEqual(
                    low,
                    high,
                    msg=f"{genre}.{field}: low {low} > high {high}",
                )

    def test_descriptions_are_non_empty_strings(self) -> None:
        """Every genre must carry a human-readable description."""
        for genre, profile in GENRE_DATABASE.items():
            self.assertIsInstance(profile["description"], str)
            self.assertTrue(
                profile["description"].strip(),
                msg=f"Genre '{genre}' has empty description",
            )


if __name__ == "__main__":
    unittest.main()
