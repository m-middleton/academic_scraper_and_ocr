import os
import re
import json
import time
import requests
import argparse
import random
import sys
import getpass
import keyring
from pathlib import Path
from urllib.parse import quote_plus, urlparse
from typing import List, Dict, Any, Optional, Tuple


class CredentialManager:
    """Manages credentials for different academic paper sources."""
    
    def __init__(self):
        self.credentials = {}
        self.service_name = "paper_scraper"
    
    def add_credentials(self, source: str, username: str, password: str, save: bool = False):
        """Add credentials for a source.
        
        Args:
            source: Name of the source (e.g., "sciencedirect", "springer")
            username: Username for the source
            password: Password for the source
            save: Whether to save the credentials securely
        """
        self.credentials[source] = {"username": username, "password": password}
        
        if save:
            try:
                keyring.set_password(self.service_name, f"{source}_{username}", password)
                print(f"Credentials for {source} saved securely.")
            except Exception as e:
                print(f"Failed to save credentials: {e}")
    
    def load_credentials(self, source: str, username: str) -> bool:
        """Load credentials for a source from secure storage.
        
        Args:
            source: Name of the source
            username: Username for the source
            
        Returns:
            True if credentials were loaded successfully
        """
        try:
            password = keyring.get_password(self.service_name, f"{source}_{username}")
            if password:
                self.credentials[source] = {"username": username, "password": password}
                return True
            return False
        except Exception as e:
            print(f"Failed to load credentials: {e}")
            return False
    
    def get_credentials(self, source: str) -> Optional[Dict[str, str]]:
        """Get credentials for a source.
        
        Args:
            source: Name of the source
            
        Returns:
            Dictionary with username and password or None if not found
        """
        return self.credentials.get(source)


class Publisher:
    """Base class for academic publishers."""
    
    def __init__(self, session: requests.Session, credential_manager: CredentialManager):
        self.session = session
        self.credential_manager = credential_manager
        self.logged_in = False
    
    def login(self) -> bool:
        """Login to the publisher site.
        
        Returns:
            True if login was successful
        """
        raise NotImplementedError("Subclasses must implement login method")
    
    def download(self, url: str, output_path: str) -> bool:
        """Download a paper from the publisher.
        
        Args:
            url: URL of the paper
            output_path: Path to save the paper
            
        Returns:
            True if download was successful
        """
        raise NotImplementedError("Subclasses must implement download method")


class ScienceDirect(Publisher):
    """Handler for Science Direct (Elsevier) papers."""
    
    def login(self) -> bool:
        """Login to Science Direct."""
        creds = self.credential_manager.get_credentials("sciencedirect")
        if not creds:
            print("No credentials found for Science Direct.")
            return False
        
        try:
            login_url = "https://id.elsevier.com/as/authorization.oauth2"
            payload = {
                "username": creds["username"],
                "password": creds["password"],
                "client_id": "SDFE-v3",
                "redirect_uri": "https://www.sciencedirect.com/"
            }
            
            response = self.session.post(login_url, data=payload)
            if response.ok and "SCIENCEDIRECT_SESSION" in self.session.cookies:
                self.logged_in = True
                print("Successfully logged in to Science Direct.")
                return True
            else:
                print("Failed to login to Science Direct.")
                return False
        except Exception as e:
            print(f"Error logging in to Science Direct: {e}")
            return False
    
    def download(self, url: str, output_path: str) -> bool:
        """Download a paper from Science Direct."""
        if not self.logged_in and not self.login():
            return False
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            # Find the PDF URL - this will depend on the site structure
            # This is a simplified example and might need adjustment
            pdf_url = None
            for pdf_attempt_url in [
                url.replace("/article/", "/article/pii/") + "/pdf",
                url + "/pdfft?isDTMRedir=true&download=true"
            ]:
                try:
                    pdf_response = self.session.get(pdf_attempt_url, stream=True)
                    if pdf_response.ok and pdf_response.headers.get('content-type') == 'application/pdf':
                        pdf_url = pdf_attempt_url
                        break
                except:
                    continue
            
            if pdf_url:
                pdf_response = self.session.get(pdf_url, stream=True)
                pdf_response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    for chunk in pdf_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                return True
            else:
                print(f"Could not find PDF link for {url}")
                return False
                
        except Exception as e:
            print(f"Error downloading from Science Direct: {e}")
            return False


class Springer(Publisher):
    """Handler for Springer papers."""
    
    def login(self) -> bool:
        """Login to Springer."""
        creds = self.credential_manager.get_credentials("springer")
        if not creds:
            print("No credentials found for Springer.")
            return False
        
        try:
            login_url = "https://link.springer.com/signup/login"
            payload = {
                "login": creds["username"],
                "password": creds["password"],
            }
            
            response = self.session.post(login_url, data=payload)
            if response.ok:
                self.logged_in = True
                print("Successfully logged in to Springer.")
                return True
            else:
                print("Failed to login to Springer.")
                return False
        except Exception as e:
            print(f"Error logging in to Springer: {e}")
            return False
    
    def download(self, url: str, output_path: str) -> bool:
        """Download a paper from Springer."""
        if not self.logged_in and not self.login():
            return False
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            # Find the PDF URL - Springer usually has a predictable URL pattern
            pdf_url = url + "/pdf"
            
            pdf_response = self.session.get(pdf_url, stream=True)
            pdf_response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in pdf_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
                
        except Exception as e:
            print(f"Error downloading from Springer: {e}")
            return False


class IEEE(Publisher):
    """Handler for IEEE papers."""
    
    def login(self) -> bool:
        """Login to IEEE."""
        creds = self.credential_manager.get_credentials("ieee")
        if not creds:
            print("No credentials found for IEEE.")
            return False
        
        try:
            login_url = "https://ieeexplore.ieee.org/servlet/Login"
            payload = {
                "username": creds["username"],
                "password": creds["password"],
            }
            
            response = self.session.post(login_url, data=payload)
            if response.ok:
                self.logged_in = True
                print("Successfully logged in to IEEE.")
                return True
            else:
                print("Failed to login to IEEE.")
                return False
        except Exception as e:
            print(f"Error logging in to IEEE: {e}")
            return False
    
    def download(self, url: str, output_path: str) -> bool:
        """Download a paper from IEEE."""
        if not self.logged_in and not self.login():
            return False
        
        try:
            article_id = url.split('/')[-1]
            
            # IEEE uses a specific PDF endpoint
            pdf_url = f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={article_id}"
            
            pdf_response = self.session.get(pdf_url, stream=True)
            pdf_response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in pdf_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
                
        except Exception as e:
            print(f"Error downloading from IEEE: {e}")
            return False


class PaperScraper:
    """A class to search and download academic papers based on keywords."""
    
    def __init__(self, output_dir: str = "papers_output"):
        """Initialize the PaperScraper.
        
        Args:
            output_dir: Directory to save downloaded papers
        """
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.semantic_scholar_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        self.scopus_api_key = os.environ.get("SCOPUS_API_KEY")
        self.wos_api_key = os.environ.get("WOS_API_KEY")
        
        if self.semantic_scholar_api_key:
            self.session.headers.update({
                "x-api-key": self.semantic_scholar_api_key
            })
        
        self.credential_manager = CredentialManager()
        
        self.publishers = {
            "sciencedirect": ScienceDirect(self.session, self.credential_manager),
            "springer": Springer(self.session, self.credential_manager),
            "ieee": IEEE(self.session, self.credential_manager)
        }
        
        self.existing_metrics = []
        self.highest_file_index = 0
        self._scan_existing_downloads()
    
    def _scan_existing_downloads(self) -> None:
        """Scan the output directory for existing downloads and metrics.
        
        This helps with resuming downloads by finding the highest numbered paper
        and loading existing metrics.
        """
        if not os.path.exists(self.output_dir):
            return
            
        paper_files = [f for f in os.listdir(self.output_dir) if f.startswith("paper_") and f.endswith(".pdf")]
        if paper_files:
            # Extract numbers from filenames (paper_00001.pdf -> 1)
            file_indices = []
            for file in paper_files:
                try:
                    number_part = re.search(r'paper_(\d+)\.pdf', file)
                    if number_part:
                        file_indices.append(int(number_part.group(1)))
                except:
                    continue
            
            if file_indices:
                self.highest_file_index = max(file_indices)
                print(f"Found existing downloads. Highest paper number: {self.highest_file_index}")
        
        metrics_file = os.path.join(self.output_dir, "paper_metrics.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    self.existing_metrics = json.load(f)
                print(f"Loaded metrics for {len(self.existing_metrics)} existing papers")
            except Exception as e:
                print(f"Error loading existing metrics: {e}")
                self.existing_metrics = []
    
    def add_credentials(self, source: str, username: str, password: str = None, save: bool = False):
        """Add credentials for a source.
        
        Args:
            source: Name of the source (e.g., "sciencedirect", "springer")
            username: Username for the source
            password: Password for the source (if None, will prompt)
            save: Whether to save the credentials securely
        """
        if password is None:
            password = getpass.getpass(f"Enter password for {username} at {source}: ")
        
        self.credential_manager.add_credentials(source, username, password, save)
    
    def search_semantic_scholar(self, keywords: List[str], paper_count: int, start_year: Optional[int] = None, end_year: Optional[int] = None, start_token: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for papers using Semantic Scholar Bulk API, supporting pagination and saving config.
        
        Args:
            keywords: List of keywords to search for
            paper_count: Number of papers to return
            start_year: Filter papers published on or after this year (inclusive)
            end_year: Filter papers published on or before this year (inclusive)
            start_token: Token to start the search from (for resuming)
            
        Returns:
            List of paper dictionaries with metadata
        """
        query = " | ".join(f'"{keyword}"' for keyword in keywords)
        
        papers = []
        max_retries = 5
        base_delay = 10
        current_token = start_token 
        
        url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
        
        fields = "title,authors,year,venue,publicationDate,url,openAccessPdf,citationCount,influentialCitationCount,referenceCount"
        
        initial_config_saved = False

        while len(papers) < paper_count:
            query_params = {
                "query": query,
                "fields": fields,
                "openAccessPdf": True
            }
            
            if start_year is not None:
                query_params["year"] = f"{start_year}-"
                if end_year is not None:
                    query_params["year"] = f"{start_year}-{end_year}"
            elif end_year is not None:
                query_params["year"] = f"-{end_year}"
                
            if current_token:
                query_params["token"] = current_token
            
            headers = {}
            if self.semantic_scholar_api_key:
                headers["x-api-key"] = self.semantic_scholar_api_key
            
            # Save initial config before first request (if not resuming)
            if not initial_config_saved and not start_token:
                 config_data = {
                    "api_used": "semantic_scholar",
                    "keywords": " ".join(keywords),
                    "desired_paper_count": paper_count, 
                    "start_year": start_year if start_year else "Not specified",
                    "end_year": end_year if end_year else "Not specified",
                    "last_used_token": "None",
                    "next_token": "None",
                    "current_papers_fetched": 0
                 }
                 self.save_config(config_data)
                 initial_config_saved = True
            
            next_token_for_batch = None

            for retry in range(max_retries):
                try:
                    # Add a small delay before each request to avoid rate limiting
                    time.sleep(1 + random.random())
                    
                    print(f"Requesting Semantic Scholar Bulk API. Current paper count: {len(papers)}. Using token: {current_token}")
                    response = self.session.get(url, params=query_params, headers=headers)
                    
                    # Handle rate limiting with exponential backoff
                    if response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', base_delay * (2 ** retry)))
                        print(f"Rate limited. Waiting {retry_after} seconds before retrying...")
                        time.sleep(retry_after)
                        continue
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    batch = data.get("data", [])
                    next_token_for_batch = data.get("token") # Get the token for the NEXT request
                    
                    if not batch:
                        print("No more papers found from Semantic Scholar.")
                        # Update config one last time to show no next token
                        config_data = {
                            "last_used_token": current_token if current_token else "None",
                            "next_token": "None", 
                            "current_papers_fetched": len(papers)
                        }
                        self.save_config(config_data)
                        return papers[:paper_count] 
                        
                    papers.extend(batch)
                    
                    print(f"Received {len(batch)} papers. Total: {len(papers)}. Next token: {next_token_for_batch}")
                    
                    # Save config after processing the batch
                    config_data = {
                        "last_used_token": current_token if current_token else "None",
                        "next_token": next_token_for_batch if next_token_for_batch else "None",
                        "current_papers_fetched": len(papers)
                    }
                    self.save_config(config_data)
                    initial_config_saved = True 

                    # If we have enough papers or there's no token for the next page, stop
                    if len(papers) >= paper_count or not next_token_for_batch:
                        if not next_token_for_batch:
                             print("Reached end of results from Semantic Scholar.")
                        return papers[:paper_count]
                    
                    current_token = next_token_for_batch
                    
                    # Break retry loop if successful
                    break 
                except requests.exceptions.RequestException as e:
                    print(f"Error during Semantic Scholar request: {e}")
                    if retry == max_retries - 1:
                        print("Max retries reached. Returning what we have.")
                        # Update config to reflect the last token that might have worked or current if none did
                        config_data = {
                            "last_used_token": current_token if current_token else "None",
                            "next_token": next_token_for_batch if next_token_for_batch else "None", # Save potential next token
                            "current_papers_fetched": len(papers)
                        }
                        self.save_config(config_data)
                        return papers[:paper_count] # Return what we have so far
        
        # Should ideally not be reached if loop condition is correct, but return just in case
        return papers[:paper_count]
    
    def search_scopus(self, keywords: List[str], paper_count: int) -> List[Dict[str, Any]]:
        """Search for papers using Scopus API.
        
        Args:
            keywords: List of keywords to search for
            paper_count: Number of papers to return
            
        Returns:
            List of paper dictionaries with metadata
        """
        if not self.scopus_api_key:
            print("Scopus API key not found. Skipping Scopus search.")
            return []
        
        query = " AND ".join(keywords)
        papers = []
        
        try:
            url = "https://api.elsevier.com/content/search/scopus"
            headers = {
                "X-ELS-APIKey": self.scopus_api_key,
                "Accept": "application/json"
            }
            params = {
                "query": query,
                "count": paper_count,
                "view": "COMPLETE"
            }
            
            response = self.session.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = data.get("search-results", {}).get("entry", [])
            
            for item in results[:paper_count]:
                paper = {
                    "title": item.get("dc:title"),
                    "authors": [{"name": author.get("authname")} for author in item.get("author", [])],
                    "year": item.get("prism:coverDate", "")[:4] if item.get("prism:coverDate") else None,
                    "venue": item.get("prism:publicationName"),
                    "url": item.get("prism:url"),
                    "citationCount": item.get("citedby-count"),
                }
                papers.append(paper)
                
        except requests.exceptions.RequestException as e:
            print(f"Error searching Scopus: {e}")
        
        return papers
    
    def search_web_of_science(self, keywords: List[str], paper_count: int) -> List[Dict[str, Any]]:
        """Search for papers using Web of Science API.
        
        Args:
            keywords: List of keywords to search for
            paper_count: Number of papers to return
            
        Returns:
            List of paper dictionaries with metadata
        """
        if not self.wos_api_key:
            print("Web of Science API key not found. Skipping WoS search.")
            return []
        
        # Implementation would depend on specific API endpoint structure
        # This is a placeholder
        print("Web of Science API implementation requires specific account details.")
        return []
    
    def download_paper(self, paper: Dict[str, Any], file_index: int) -> Optional[str]:
        """Download a paper and save it to the output directory with a sequential filename.
        
        Args:
            paper: Paper metadata dictionary
            file_index: Sequential index for naming the file
            
        Returns:
            Path to the downloaded file or None if download failed
        """
        title = paper.get("title", "Unknown Title")
        
        filename = f"paper_{file_index:05d}.pdf"
        file_path = os.path.join(self.output_dir, filename)
        
        if paper.get("openAccessPdf") and paper["openAccessPdf"].get("url"):
            pdf_url = paper["openAccessPdf"]["url"]
            
            try:
                # Add a small delay before downloading to avoid rate limiting
                time.sleep(1 + random.random())
                
                response = self.session.get(pdf_url, stream=True)
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"Downloaded open access PDF: '{title}' as {filename}")
                return file_path
                
            except requests.exceptions.RequestException as e:
                print(f"Error downloading open access PDF '{title}': {e}")
                # Continue to try other methods
        
        if paper.get("url"):
            url = paper["url"]
            domain = urlparse(url).netloc
            
            publisher = None
            if "sciencedirect.com" in domain or "elsevier.com" in domain:
                publisher = self.publishers.get("sciencedirect")
            elif "springer.com" in domain or "link.springer.com" in domain:
                publisher = self.publishers.get("springer")
            elif "ieee.org" in domain or "ieeexplore.ieee.org" in domain:
                publisher = self.publishers.get("ieee")
            
            if publisher:
                if publisher.download(url, file_path):
                    print(f"Downloaded via publisher access: '{title}' as {filename}")
                    return file_path
        
        print(f"No PDF available for: '{title}'")
        return None
    
    def extract_metrics(self, paper: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Extract metrics for a paper, including the filename.
        
        Args:
            paper: Paper metadata dictionary
            filename: The filename the paper was saved as
            
        Returns:
            Dictionary of paper metrics
        """
        pub_type = "unknown"
        
        if paper.get("venue") == "arXiv":
            pub_type = "preprint"
        elif paper.get("venue"):
            venue_lower = paper.get("venue", "").lower()
            if any(conf_term in venue_lower for conf_term in ["conference", "proc.", "proceedings", "symposium", "workshop"]):
                pub_type = "conference"
            elif any(journal_term in venue_lower for journal_term in ["journal", "transactions", "review", "letters"]):
                pub_type = "journal"
        
        metrics = {
            "title": paper.get("title", "Unknown Title"),
            "filename": filename,
            "authors": [author.get("name", "") for author in paper.get("authors", [])],
            "year": paper.get("year"),
            "venue": paper.get("venue"),
            "publication_type": pub_type,
            "citation_count": paper.get("citationCount"),
            "influential_citation_count": paper.get("influentialCitationCount"),
            "reference_count": paper.get("referenceCount"),
            "url": paper.get("url")
        }
        
        return metrics
    
    def save_metrics(self, metrics: List[Dict[str, Any]]) -> None:
        """Save paper metrics to a JSON file.
        
        Args:
            metrics: List of paper metrics dictionaries
        """
        metrics_file = os.path.join(self.output_dir, "paper_metrics.json")
        metrics_temp_file = os.path.join(self.output_dir, "paper_metrics.json.tmp")
        
        try:
            # First write to a temporary file
            with open(metrics_temp_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
                
            # Then rename it to the actual file (atomic operation)
            # This prevents corrupt files if the process is interrupted during writing
            import shutil
            shutil.move(metrics_temp_file, metrics_file)
        except Exception as e:
            print(f"Error saving metrics file: {e}")
            # If temporary file exists but failed to be moved, try to clean it up
            if os.path.exists(metrics_temp_file):
                try:
                    os.remove(metrics_temp_file)
                except:
                    pass

    def save_config(self, config_data: Dict[str, Any]) -> None:
        """Save or update the run configuration to a text file.

        Args:
            config_data: Dictionary containing configuration parameters.
        """
        config_file = os.path.join(self.output_dir, "config.txt")
        
        existing_config = {}
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if ":" in line:
                            key, value = line.strip().split(":", 1)
                            existing_config[key.strip()] = value.strip()
            except Exception as e:
                print(f"Warning: Could not read existing config file: {e}")
        
        existing_config.update(config_data)
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                for key, value in existing_config.items():
                    f.write(f"{key}: {value}\n")
            print(f"Updated configuration in {config_file}")
        except Exception as e:
            print(f"Error writing config file: {e}")
    
    def search_arxiv(self, keywords: List[str], paper_count: int) -> List[Dict[str, Any]]:
        """Search for papers using arXiv API as a fallback.
        
        Args:
            keywords: List of keywords to search for
            paper_count: Number of papers to return
            
        Returns:
            List of paper dictionaries with metadata
        """
        try:
            import feedparser
            
            query_string = " OR ".join([f'all:{quote_plus(kw)}' for kw in keywords])
            
            search = arxiv.Search(
                query=query_string,
                max_results=paper_count,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            for entry in search.results():
                pdf_url = None
                for link in entry.links:
                    if link.title == 'pdf':
                        pdf_url = link.href
                        break
                
                authors = [{"name": author.name} for author in entry.authors]
                
                paper = {
                    "title": entry.title,
                    "authors": authors,
                    "year": entry.published[:4] if hasattr(entry, 'published') else None,
                    "venue": "arXiv", # arXiv doesn't have a traditional venue
                    "url": entry.link,
                    "openAccessPdf": {"url": pdf_url} if pdf_url else None,
                    "citationCount": None,  # arXiv doesn't provide citation counts
                    "influentialCitationCount": None,
                    "referenceCount": None
                }
                
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"Error searching arXiv: {e}")
            return []
    
    def process_keywords(self, keywords: List[str], paper_count: int, api: str = "semantic_scholar", start_year: Optional[int] = None, end_year: Optional[int] = None, start_token: Optional[str] = None) -> Tuple[int, List[Dict[str, Any]]]:
        """Search for papers, download them, and extract metrics.
        
        Args:
            keywords: List of keywords to search for
            paper_count: Number of papers to return
            api: API to use for search (semantic_scholar, scopus, wos, or arxiv)
            start_year: Filter papers published on or after this year (inclusive)
            end_year: Filter papers published on or before this year (inclusive)
            start_token: Token to start the Semantic Scholar search from (for resuming)
            
        Returns:
            A tuple containing:
                - Number of downloaded papers
                - List of paper metrics
        """
        resume_download = False
        papers_to_skip = 0
        
        if self.highest_file_index > 0:
            resume_download = True
            papers_to_skip = self.highest_file_index
            print(f"Resuming download from paper #{papers_to_skip + 1}. Will download {paper_count - papers_to_skip} more papers.")
            
            if papers_to_skip >= paper_count:
                print(f"Already have {papers_to_skip} papers, which meets or exceeds the requested count of {paper_count}.")
                return papers_to_skip, self.existing_metrics
        
        year_range_str = ""
        if start_year and end_year:
            year_range_str = f" from years {start_year}-{end_year}"
        elif start_year:
            year_range_str = f" from year {start_year} onwards"
        elif end_year:
            year_range_str = f" until year {end_year}"
            
        print(f"Searching for {paper_count} papers with keywords: {keywords}{year_range_str}")
        
        papers = []
        
        if api == "scopus" and self.scopus_api_key:
            # Note: Scopus/WoS/ArXiv don't support token resumption in this script
            if start_token:
                print("Warning: --start-token is only supported for Semantic Scholar API.")
            papers = self.search_scopus(keywords, paper_count)
        elif api == "wos" and self.wos_api_key:
            if start_token:
                print("Warning: --start-token is only supported for Semantic Scholar API.")
            papers = self.search_web_of_science(keywords, paper_count)
        elif api == "arxiv":
            if start_token:
                print("Warning: --start-token is only supported for Semantic Scholar API.")
            papers = self.search_arxiv(keywords, paper_count)
        else: 
            actual_paper_count = paper_count
            # If resuming, we need to get more papers than just the remaining count
            if resume_download:
                # Get the full set and we'll skip as needed
                actual_paper_count = paper_count 
            papers = self.search_semantic_scholar(keywords, actual_paper_count, start_year, end_year, start_token)
        
        # Try fallback options if no papers found (currently only from Semantic Scholar)
        if not papers and api == "semantic_scholar":
            print("No papers found on Semantic Scholar. Trying arXiv as fallback...")
            try:
                import feedparser # Import here to avoid making it a hard dependency
                import arxiv # Import here for the same reason
                papers = self.search_arxiv(keywords, paper_count)
            except ImportError:
                print("Feedparser or arxiv library not installed. Cannot use arXiv fallback.")
                print("Install with: pip install feedparser arxiv")
        
        if not papers:
            print("No papers found matching the search criteria.")
            if resume_download and self.existing_metrics:
                return papers_to_skip, self.existing_metrics
            return 0, []
        
        print(f"Processing {len(papers)} papers.")
        
        # If resuming, skip papers we already have
        if resume_download and papers_to_skip > 0:
            # We'll skip the first papers_to_skip papers since we already have them
            print(f"Skipping first {papers_to_skip} papers that were already downloaded.")
            if papers_to_skip < len(papers):
                papers = papers[papers_to_skip:]
            else:
                # If we already have more papers than what we got from search
                print("Already have more papers than what was returned by the search.")
                return papers_to_skip, self.existing_metrics
        
        downloaded_count = 0
        new_metrics = []
        
        start_file_index = self.highest_file_index + 1
        
        for i, paper in enumerate(papers):
            file_index = start_file_index + i
            file_path = self.download_paper(paper, file_index)
            
            # Only extract and save metrics if download was successful
            if file_path:
                filename = os.path.basename(file_path)
                metrics = self.extract_metrics(paper, filename)
                new_metrics.append(metrics)
                downloaded_count += 1
                
                # Save metrics after each successful download
                all_metrics = self.existing_metrics + new_metrics
                self.save_metrics(all_metrics)
                print(f"Downloaded paper {file_index}. Updated metrics file ({len(all_metrics)} total papers).")
            
            # Be nice to the server
            time.sleep(1 + random.random())
        
        all_metrics = self.existing_metrics + new_metrics
        
        if not all_metrics:
            print("No papers were downloaded, no metrics saved")
        
        total_downloaded = papers_to_skip + downloaded_count
        return total_downloaded, all_metrics


def main():
    parser = argparse.ArgumentParser(description="Search and download academic papers based on keywords")
    parser.add_argument("--count", type=int, default=10, help="Number of papers to return")
    parser.add_argument("--api", choices=["semantic_scholar", "scopus", "wos", "arxiv"], default="semantic_scholar",
                        help="API to use for search")
    parser.add_argument("--output", default="papers_output", help="Directory to save downloaded papers")
    parser.add_argument("--start-year", type=int, help="Filter papers published on or after this year")
    parser.add_argument("--end-year", type=int, help="Filter papers published on or before this year")
    parser.add_argument("--start-token", type=str, help="Semantic Scholar token to resume search from")
    parser.add_argument("--login", action="store_true", help="Add login credentials for publisher access")
    parser.add_argument("--source", choices=["sciencedirect", "springer", "ieee"], 
                        help="Publisher to add credentials for")
    parser.add_argument("--username", help="Username for publisher login")
    parser.add_argument("--save-credentials", action="store_true", help="Save credentials securely for future use")
    parser.add_argument("--keyword-set", choices=["big_data", "glacial_hydrology", "religious_studies"], 
                        help="Specific keyword set to process instead of all sets")
    parser.add_argument("--year-range", help="Specific year range to process instead of all ranges (e.g., '2022-2027')")
    
    args = parser.parse_args()
    
    # Install feedparser if using arxiv API and not already installed
    if args.api == "arxiv":
        try:
            import feedparser
            import arxiv # Added arxiv import here as well
        except ImportError:
            print("Installing feedparser and arxiv for arXiv API support...")
            import subprocess
            subprocess.call([sys.executable, "-m", "pip", "install", "feedparser", "arxiv"])
            print("Please re-run the script after installation.")
            sys.exit(0) # Exit after attempting install as libraries might not be immediately available
    
    # Install keyring if saving credentials
    if args.save_credentials:
        try:
            import keyring
        except ImportError:
            print("Installing keyring for secure credential storage...")
            import subprocess
            subprocess.call([sys.executable, "-m", "pip", "install", "keyring"])
            print("Please re-run the script after installation.")
            sys.exit(0) # Exit after attempting install
    
    scraper = PaperScraper(output_dir=args.output)
    
    if args.login:
        if not args.source:
            print("Please specify a source with --source")
            return
        
        if not args.username:
            args.username = input(f"Enter username for {args.source}: ")
        
        password = getpass.getpass(f"Enter password for {args.username} at {args.source}: ")
        scraper.add_credentials(args.source, args.username, password, args.save_credentials)
        print(f"Credentials added for {args.source}")
        # If only logging in, exit here
        # Check if any search-related args were passed along with --login
        search_args_present = any([args.count != 10, args.api != "semantic_scholar", args.start_year, args.end_year, args.start_token, args.keyword_set, args.year_range])
        # Check if hardcoded keywords (which would be used by default if no keyword_set is given) are intended for processing.
        # This assumes that if keywords are not explicitly provided via CLI (e.g. through a --keywords arg not present here), 
        # and no other search parameters are changed from defaults, then login might be the sole action.
        # For this logic to be robust, it depends on how keyword_sets are used later if not specified.
        # A specific --keywords argument would simplify this check.
        if not search_args_present:
             # Check if default keyword processing is implicitly expected.
             # This is tricky. Assuming for now if no search params changed, only login was intended.
             # A clearer signal (like a specific --run-search flag) would be better.
             print("Login action complete. No other search parameters specified. Exiting.")
             return
         
    big_data_keywords = [
        "Big Data Analytics",
        "Data Mining",
        "Machine Learning",
        "Cloud Computing",
        "Distributed Systems",
        "Hadoop",
        "Apache Spark",
        "NoSQL Databases",
        "Data Processing Frameworks",
        "Real-time Data Processing",
        "Stream Analytics",
        "Data Visualization",
        "Data Security",
        "Data Privacy",
        "Data Governance",
        "Scalable Data Architectures",
        "Data Lakes",
        "Predictive Modeling",
        "Data Integration",
        "Big Data Infrastructure"
    ]

    glacial_keywords = [
        "Glacier Hydrology",
        "Supraglacial Hydrology",
        "Englacial Hydrology",
        "Subglacial Hydrology",
        "Meltwater Runoff",
        "Glacier Mass Balance",
        "Glacier Dynamics",
        "Ice Sheet Hydrology",
        "Proglacial Lakes",
        "Glacier Outburst Floods (GLOFs)",
        "Glacier Hydrochemistry",
        "Isotope Hydrology (Glaciers)",
        "Ground Penetrating Radar (GPR) Glaciology",
        "Moulin",
        "Subglacial Drainage Systems",
        "Glacier Retreat Impact",
        "Climate Change Glacier Hydrology",
        "Remote Sensing Glaciology",
        "Snowmelt Hydrology",
        "Cryosphere Hydrology"
    ]

    religious_studies_keywords = [
        "Biblical Exegesis",
        "Hermeneutics",
        "Textual Criticism (Bible)",
        "Old Testament / Hebrew Bible",
        "New Testament Studies",
        "Biblical Hebrew",
        "Koine Greek",
        "Biblical Aramaic",
        "Septuagint (LXX)",
        "Dead Sea Scrolls",
        "Biblical Archaeology",
        "Ancient Near East Context",
        "Greco-Roman Context",
        "Second Temple Judaism",
        "Early Christianity Studies",
        "Gospels Research",
        "Pauline Epistles",
        "Biblical Theology",
        "Literary Criticism (Bible)",
        "Social-Scientific Criticism (Bible)"
    ]

    loop_dict = {
        "religious_studies": {
            "keywords": religious_studies_keywords,
            "years": [("2000-2022", 1000),  ("2022-2027", 1000)] # (years, paper count)
        }
    }

    # If specific keyword set is requested, only process that one
    if args.keyword_set:
        if args.keyword_set in loop_dict:
            keyword_sets_to_process = {args.keyword_set: loop_dict[args.keyword_set]}
        else:
            print(f"Unknown keyword set: {args.keyword_set}")
            return
    else:
        keyword_sets_to_process = loop_dict

    initial_scraper_credentials = scraper.credential_manager # Store credentials from the initial scraper

    for keyword_set_name, keyword_set in keyword_sets_to_process.items():
        keywords = keyword_set["keywords"]
        years_to_process = keyword_set["years"]
        
        # If specific year range is requested, only process that one
        if args.year_range:
            years_to_process = [(yr_range, paper_count) for yr_range, paper_count in years_to_process if yr_range == args.year_range]
            if not years_to_process:
                print(f"Year range {args.year_range} not found for keyword set {keyword_set_name}")
                continue

        for year_range, paper_count in years_to_process:
            try:
                start_year_str, end_year_str = year_range.split("-")
                
                start_year = int(start_year_str) if start_year_str else None
                end_year = int(end_year_str) if end_year_str else None
            except ValueError:
                print(f"Invalid year range format: {year_range}, skipping...")
                continue

            output_path = os.path.join(args.output, f"{keyword_set_name}_{year_range}")
            print(f"\nProcessing keyword set '{keyword_set_name}' for years {year_range}")
            print(f"Output directory: {output_path}")
            
            # Create a new scraper with this specific output path
            # This will automatically check for existing downloads
            current_run_scraper = PaperScraper(output_dir=output_path)
            
            # Transfer credentials from the initial scraper instance if they were set via --login
            for source_name in ["sciencedirect", "springer", "ieee"]:
                creds = initial_scraper_credentials.get_credentials(source_name)
                if creds:
                    current_run_scraper.add_credentials(source_name, creds["username"], creds["password"], False)
            
            downloaded_count, metrics = current_run_scraper.process_keywords(
                keywords, 
                paper_count, 
                args.api,
                start_year,
                end_year,
                args.start_token 
            )
            
            print(f"\nDownloaded {downloaded_count} papers in this run to {output_path}")

if __name__ == "__main__":
    import sys
    main()
