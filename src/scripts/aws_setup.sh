#!/bin/bash

# Update system packages
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and required packages
sudo apt-get install -y python3 python3-pip python3-venv python3-full

# Install required system packages
sudo apt-get install -y git

# Create directory for the application
mkdir -p ~/rain
cd ~/rain

# Clone the repository (replace with your actual repo URL)
git clone https://github.com/sid-chava/Rain.git .

# Create .env file
echo "Creating .env file..."
echo "Please enter your environment variables:"

read -p "SUPABASE_URL: " supabase_url
read -p "SUPABASE_SERVICE_KEY: " supabase_key
read -p "OPENAI_API_KEY: " openai_key

cat > .env << EOL
SUPABASE_URL=${supabase_url}
SUPABASE_SERVICE_KEY=${supabase_key}
OPENAI_API_KEY=${openai_key}
GMAIL_CREDENTIALS_PATH=credentials.json
EOL

echo ".env file created successfully!"

# Remove any existing venv
rm -rf venv

# Create and activate virtual environment
python3 -m venv venv
. venv/bin/activate

# Upgrade pip in the virtual environment
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Create directory for logs
mkdir -p logs

# Create the cron job script
cat > run_dedup_job.sh << 'EOL'
#!/bin/bash
cd ~/rain
. venv/bin/activate
python -m src.scripts.deduplication_job >> logs/dedup_job.log 2>&1
EOL

# Make the script executable
chmod +x run_dedup_job.sh

# Add cron job to run every 6 hours
(crontab -l 2>/dev/null; echo "0 */6 * * * ~/rain/run_dedup_job.sh") | crontab -

# Create script to check logs
cat > check_logs.sh << 'EOL'
#!/bin/bash
tail -f logs/dedup_job.log
EOL

chmod +x check_logs.sh

echo "Setup completed! The deduplication job will run every 6 hours."
echo "To check the logs, run: ./check_logs.sh"
echo ""
echo "IMPORTANT: Don't forget to copy your Gmail credentials file (credentials.json) to ~/rain/ if you're using Gmail ingestion!"
echo "You can do this using: scp -i rain-key.pem credentials.json ubuntu@3.138.174.94:~/rain/" 

scp -i rain-key.pem credentials.json ubuntu@3.138.174.94:~/rain/