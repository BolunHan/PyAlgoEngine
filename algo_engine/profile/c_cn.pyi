from .c_base import Profile


class ProfileCN(Profile):
    """
    China A stock market profile. use override_profile() to set as default profile.

    Note that the override_profile() method will also set the system timezone of this process to Asia/Shanghai.
    """
