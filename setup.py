from setuptools import setup

setup(
    name="medibot",
    version="1.0",
    install_requires=[
        "flask",
        "pytz",
        "openai",
        # Do not include pywin32 if you're not deploying to Windows
    ],
)
