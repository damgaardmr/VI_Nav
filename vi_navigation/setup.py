from setuptools import setup

package_name = 'vi_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Malte R. Damgaard',
    maintainer_email='DamgaardMR@gmail.com',
    description='This package implements different probabilistic navigation algorithms for non-holonomic uni-cycle robots using variational inference.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
            'console_scripts': [
                    'shuttle_bus_navigation = vi_navigation.shuttle_bus_navigation:main',
                    'shuttle_bus_navigation_cooperative = vi_navigation.shuttle_bus_navigation_cooperative:main',
                    'shuttle_bus_navigation_cooperative_guassian_msg = vi_navigation.shuttle_bus_navigation_cooperative_guassian_msg:main',
            ],
    },
)
