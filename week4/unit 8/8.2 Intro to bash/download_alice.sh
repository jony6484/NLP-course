mkdir alice
echo "created the alice dir"
curl https://gist.githubusercontent.com/phillipj/4944029/raw/75ba2243dd5ec2875f629bf5d79f6c1e4b5a8b46/alice_in_wonderland.txt --output ./alice/alice_book.txt
echo "Downloaded alice_book.txt"
cp ./alice/alice_book.txt ./alice/alice.txt
echo "Renamed alice_book.txt to alice.txt"
echo "First 3 line of the book:"
head -3 ./alice/alice.txt


