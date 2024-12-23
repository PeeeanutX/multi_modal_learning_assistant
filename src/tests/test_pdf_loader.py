import subprocess
import sys
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PipManager:
    def __init__(self):
        pass

    @staticmethod
    def install_with_extra_index(packages: list, extra_index_url: str):
        """
        Installs Python packages using pip with an extra index URL.
        :param packages: List of packages to install.
        :param extra_index_url: The extra index URL to use.
        """
        try:
            logger.info(f"Installing packages: {', '.join(packages)} with extra index URL: {extra_index_url}")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", *packages, "--extra-index-url", extra_index_url]
            )
            logger.info(f"Successfully installed packages: {', '.join(packages)}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install packages: {', '.join(packages)}. Error: {e}")

    @staticmethod
    def uninstall(packages: list):
        """
        Uninstalls Python packages using pip.
        :param packages: List of packages to uninstall.
        """
        try:
            logger.info(f"Uninstalling packages: {', '.join(packages)}")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "uninstall", "-y", *packages]
            )
            logger.info(f"Successfully uninstalled packages: {', '.join(packages)}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to uninstall packages: {', '.join(packages)}. Error: {e}")

# Example Usage
if __name__ == "__main__":
    manager = PipManager()
    packages_to_manage = ["torch", "torchvision", "torchaudio"]
    extra_index = "https://download.pytorch.org/whl/cu118"

    # Uninstall the packages
    manager.uninstall(packages_to_manage)

    # Reinstall the packages
    manager.install_with_extra_index(packages_to_manage, extra_index)
