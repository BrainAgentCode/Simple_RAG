"""
Chrome WebDriver 统一配置（适用于 Linux 服务器 / Docker / 无图形界面环境）
"""

import logging
import os
import shutil

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

logger = logging.getLogger(__name__)


def create_chrome_options(headless: bool = True) -> Options:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--remote-allow-origins=*")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    chrome_bin = os.getenv("CHROME_BIN", "").strip()
    if not chrome_bin:
        for candidate in (
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
        ):
            if os.path.isfile(candidate):
                chrome_bin = candidate
                break
    if chrome_bin:
        options.binary_location = chrome_bin
        logger.info(f"使用 Chrome 可执行文件: {chrome_bin}")
    else:
        logger.warning(
            "未找到 Chrome/Chromium。请安装后设置 CHROME_BIN，例如: "
            "apt install chromium-browser chromium-driver"
        )

    return options


def create_chrome_driver(headless: bool = True):
    """创建 Chrome WebDriver，优先 Selenium Manager，失败则尝试 webdriver-manager。"""
    options = create_chrome_options(headless=headless)

    try:
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(60)
        return driver
    except Exception as first_error:
        logger.warning(f"Selenium Manager 启动 Chrome 失败: {first_error}")

    try:
        from webdriver_manager.chrome import ChromeDriverManager

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(60)
        return driver
    except Exception as second_error:
        chromedriver = shutil.which("chromedriver")
        if chromedriver:
            service = Service(chromedriver)
            driver = webdriver.Chrome(service=service, options=options)
            driver.set_page_load_timeout(60)
            return driver
        raise RuntimeError(
            "无法启动 Chrome。请安装 chromium 与 chromedriver，并可选设置 CHROME_BIN。\n"
            f"原始错误: {first_error}\n备用错误: {second_error}"
        ) from second_error
