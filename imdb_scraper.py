import time
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

class IMDbScraper:
    def __init__(self):
        """Initialize the scraper with Chrome options"""
        chrome_options = Options()
        # Uncomment next line to run headless (faster, no browser window)
        # chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.maximize_window()
        
        # Professional wait strategy: up to 30 seconds for slow connections
        self.wait = WebDriverWait(self.driver, 30)
        
        self.movies_data = []
        
    def get_movie_urls(self, start_page=1, max_pages=10):
        """
        Extract movie URLs from search results
        IMDb shows 50 movies per page
        """
        movie_urls = []
        
        for page in range(start_page, start_page + max_pages):
            start_index = (page - 1) * 50 + 1
            url = f'https://www.imdb.com/search/title/?title_type=feature&release_date=2024-01-01,2024-12-31&sort=release_date,asc&start={start_index}'
            
            print(f"\n🔍 Scraping page {page} (movies {start_index}-{start_index+49})...")
            
            try:
                self.driver.get(url)
                
                # Wait for movie cards to load
                self.wait.until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'li.ipc-metadata-list-summary-item'))
                )
                
                # Add small delay to ensure JavaScript finishes rendering
                time.sleep(2)
                
                # Find all movie cards
                movie_cards = self.driver.find_elements(By.CSS_SELECTOR, 'li.ipc-metadata-list-summary-item')
                
                print(f"✅ Found {len(movie_cards)} movies on this page")
                
                for card in movie_cards:
                    try:
                        # Extract the movie URL
                        link_element = card.find_element(By.CSS_SELECTOR, 'a.ipc-title-link-wrapper')
                        movie_url = link_element.get_attribute('href')
                        
                        if movie_url and '/title/tt' in movie_url:
                            # Clean URL (remove query parameters)
                            clean_url = movie_url.split('?')[0]
                            movie_urls.append(clean_url)
                            
                    except NoSuchElementException:
                        continue
                
                # Check if there are more pages
                try:
                    next_button = self.driver.find_element(By.CSS_SELECTOR, 'button[aria-label="Next"]')
                    if 'disabled' in next_button.get_attribute('class'):
                        print("\n✓ Reached last page of results")
                        break
                except:
                    pass
                    
            except TimeoutException:
                print(f"⚠️  Timeout on page {page} - retrying once...")
                time.sleep(5)
                try:
                    self.driver.get(url)
                    time.sleep(3)
                except:
                    print(f"❌ Failed to load page {page} after retry")
                    continue
                    
        print(f"\n📊 Total movies found: {len(movie_urls)}")
        return movie_urls
    
    def scrape_movie_details(self, movie_url):
        """
        Scrape title and plot summary from individual movie page
        Returns: dict with title and plot_summary
        """
        try:
            self.driver.get(movie_url)
            # give page a moment
            time.sleep(1)
            
            # ------- Title extraction (try multiple selectors, then meta fallback) -------
            title = None
            title_selectors = [
                'h1[data-testid="hero-title-block__title"]',
                'h1[data-testid="hero__pageTitle"]',
                'div.title_wrapper > h1'
            ]
            for sel in title_selectors:
                try:
                    elem = WebDriverWait(self.driver, 5).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, sel))
                    )
                    title = elem.text.strip()
                    if title:
                        break
                except:
                    continue
            
            if not title:
                # fallback to meta og:title
                try:
                    title_meta = self.driver.find_element(By.XPATH, "//meta[@property='og:title']").get_attribute('content')
                    if title_meta:
                        # often formatted like "Movie Name - IMDb"; take left part
                        title = title_meta.split('-')[0].strip()
                except:
                    title = "N/A"
            
            # ------- Plot summary extraction (prefer og:description, then selectors, then meta description) -------
            plot = "N/A"
            try:
                og_desc = self.driver.find_element(By.XPATH, "//meta[@property='og:description']").get_attribute('content')
                if og_desc and len(og_desc) > 10:
                    plot = og_desc.strip()
            except:
                pass
            
            # Try targeted selectors only if og:description didn't give a good result
            if plot == "N/A" or len(plot) < 10:
                plot_selectors = [
                    'span[data-testid="plot-l"]',
                    'span[data-testid="plot-xs_to_m"]',
                    'span[data-testid="plot-xl"]',
                    'div[data-testid="storyline-plot-summary"] p',
                    'div[data-testid="storyline"] p',
                    'div.ipc-html-content.ipc-html-content--base > p',  # generic storyline paragraph
                ]
                for sel in plot_selectors:
                    try:
                        elem = WebDriverWait(self.driver, 5).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, sel))
                        )
                        text = elem.text.strip()
                        # skip very short or obviously non-plot text
                        if text and len(text) > 10 and "User reviews" not in text:
                            plot = text
                            break
                    except:
                        continue
            
            # final fallback: meta name="description"
            if plot == "N/A" or len(plot) < 10:
                try:
                    meta_desc = self.driver.find_element(By.XPATH, "//meta[@name='description']").get_attribute('content')
                    if meta_desc and len(meta_desc) > 10:
                        plot = meta_desc.strip()
                except:
                    pass
            
            print(f"✓ Scraped: {title}")
            
            return {
                'title': title,
                'plot_summary': plot,
                'url': movie_url
            }
            
        except TimeoutException:
            print(f"⚠️  Timeout for {movie_url}")
            return None
        except Exception as e:
            print(f"❌ Error scraping {movie_url}: {str(e)}")
            return None
    
    def scrape_all_movies(self, max_pages=10):
        """Main scraping workflow"""
        print("🎬 Starting IMDb Movie Scraper for 2024")
        print("=" * 60)
        
        # Step 1: Get all movie URLs
        movie_urls = self.get_movie_urls(start_page=1, max_pages=max_pages)
        
        if not movie_urls:
            print("❌ No movies found. Exiting...")
            return
        
        # Step 2: Scrape each movie's details
        print(f"\n🎯 Scraping details for {len(movie_urls)} movies...")
        print("=" * 60)
        
        for i, url in enumerate(movie_urls, 1):
            print(f"\n[{i}/{len(movie_urls)}] Processing: {url}")
            
            movie_data = self.scrape_movie_details(url)
            
            if movie_data:
                self.movies_data.append(movie_data)
            
            # Respectful scraping: small delay between requests
            time.sleep(1)
            
            # Save progress every 50 movies
            if i % 50 == 0:
                self.save_to_csv(f'imdb_2024_movies_progress_{i}.csv')
                print(f"\n💾 Progress saved: {i} movies scraped")
        
        # Final save
        self.save_to_csv('imdb_2024_movies_final.csv')
        print(f"\n✅ Scraping complete! Total movies: {len(self.movies_data)}")
    
    def save_to_csv(self, filename='imdb_2024_movies.csv'):
        """Save scraped data to CSV file"""
        if not self.movies_data:
            print("No data to save")
            return
            
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['title', 'plot_summary', 'url']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for movie in self.movies_data:
                    writer.writerow(movie)
            
            print(f"✅ Data saved to {filename}")
            
        except Exception as e:
            print(f"❌ Error saving to CSV: {str(e)}")
    
    def close(self):
        """Close browser"""
        self.driver.quit()
        print("\n🔒 Browser closed")


# Main execution
if __name__ == "__main__":
    scraper = IMDbScraper()
    
    try:
        # Scrape first 5 pages (250 movies) - adjust max_pages as needed
        # For ALL 2024 movies, increase max_pages to 15-20
        scraper.scrape_all_movies(max_pages=1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Scraping interrupted by user")
        scraper.save_to_csv('imdb_2024_movies_interrupted.csv')
        
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        
    finally:
        scraper.close()
        print("\n🎉 Script execution completed!")
